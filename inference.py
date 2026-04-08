from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from openai import OpenAI

from env.models import Action, Observation
from tasks.easy_task import TASK_NAME as EASY_TASK_NAME
from tasks.hard_task import TASK_NAME as HARD_TASK_NAME
from tasks.medium_task import TASK_NAME as MEDIUM_TASK_NAME
from tasks import create_easy_env, create_hard_env, create_medium_env
from graders import grade_easy, grade_hard, grade_medium

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
# Backward-compatible fallback for local testing.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional when using from_docker_image() in hosted runners.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS = {
    "easy": 20,
    "medium": 25,
    "hard": 30,
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Dict[str, object], reward: float, done: bool, error: str | None = None) -> None:
    safe_action = json.dumps(action, ensure_ascii=True)
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.4f} done={str(done).lower()} error={error or 'none'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def get_model_hint(client: OpenAI, observation: Observation, task_mode: str) -> str:
    """Use OpenAI client for task reasoning; fallback to heuristic on API failure."""
    if not observation.current_email:
        return "done"

    email = observation.current_email
    prompt = (
        "You are an assistant in an email triage system. "
        f"Task mode: {task_mode}. "
        f"Sender: {email.sender}. Subject: {email.subject}. Content: {email.content}. "
        "Return a short JSON-like hint for action reasoning."
    )

    if not (HF_TOKEN or OPENAI_API_KEY):
        return "heuristic"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": "You produce concise operation hints."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
        )
        return response.choices[0].message.content or "heuristic"
    except Exception:
        return "heuristic"


def choose_action(observation: Observation, task_mode: str, hint: str) -> Action:
    if not observation.current_email:
        return Action(action_type="classify_email", classification="normal")

    email = observation.current_email
    content = (email.subject + " " + email.content + " " + hint).lower()

    if "verify" in content or "coupon" in content or email.spam_probability >= 0.7:
        label = "spam"
    elif email.urgency_level >= 4:
        label = "important"
    else:
        label = "normal"

    if label == "spam":
        reply_text = "This looks like spam/phishing. I will ignore and report it for security review."
        priority_rank = 3
    elif label == "important":
        reply_text = "Acknowledged. I will send the requested board/status update with milestones and blockers soon."
        priority_rank = 1
    else:
        reply_text = "Thanks for the update. I will acknowledge the policy request by end of week."
        priority_rank = 2

    current_state = next(
        (item for item in observation.inbox_state if item.email.email_id == email.email_id),
        None,
    )

    if task_mode == "easy":
        return Action(action_type="classify_email", classification=label)

    if task_mode == "medium":
        return Action(action_type="reply_email", reply_text=reply_text)

    # Hard mode: complete classify -> prioritize -> reply sequence per email.
    if current_state and not current_state.classified:
        return Action(action_type="classify_email", classification=label)
    if current_state and not current_state.prioritized:
        return Action(action_type="prioritize_email", priority_rank=priority_rank)
    return Action(action_type="reply_email", reply_text=reply_text)


def run_single_task(task_mode: str, task_name: str, client: OpenAI) -> Tuple[float, Dict[str, float]]:
    if task_mode == "easy":
        env = create_easy_env(seed=42)
    elif task_mode == "medium":
        env = create_medium_env(seed=42)
    else:
        env = create_hard_env(seed=42)

    rewards: List[float] = []
    score = 0.0
    steps_taken = 0

    log_start(task=task_name, env="SmartEmailTriageEnv", model=MODEL_NAME)

    try:
        observation = env.reset()
        for step in range(1, MAX_STEPS[task_mode] + 1):
            hint = get_model_hint(client, observation, task_mode)
            action = choose_action(observation, task_mode, hint)
            result = env.step(action)

            reward_value = result.reward.value
            rewards.append(reward_value)
            steps_taken = step

            log_step(
                step=step,
                action=action.model_dump(),
                reward=reward_value,
                done=result.done,
                error=None,
            )

            observation = result.observation
            if result.done:
                break

        if task_mode == "easy":
            details = grade_easy(env)
        elif task_mode == "medium":
            details = grade_medium(env)
        else:
            details = grade_hard(env)

        score = float(details["score"])
        success = score >= 0.6
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        return score, details
    finally:
        env.close()


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or OPENAI_API_KEY or "dummy-key")

    task_plan = [
        ("easy", EASY_TASK_NAME),
        ("medium", MEDIUM_TASK_NAME),
        ("hard", HARD_TASK_NAME),
    ]

    all_scores: List[float] = []
    details_bundle: Dict[str, Dict[str, float]] = {}

    for task_mode, task_name in task_plan:
        score, details = run_single_task(task_mode, task_name, client)
        all_scores.append(score)
        details_bundle[task_name] = details

    final_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    final_score = max(0.0, min(1.0, final_score))

    print("\n=== BASELINE SUMMARY ===", flush=True)
    print(f"Final aggregate score: {final_score:.4f}", flush=True)
    print(json.dumps(details_bundle, indent=2), flush=True)


if __name__ == "__main__":
    main()
