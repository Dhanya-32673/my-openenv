from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .models import Action, Email, InboxItemState, Observation, Reward, StepResult

TaskMode = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class EmailGold:
    classification: str
    reply_keywords: Tuple[str, ...]
    priority_rank: int


class SmartEmailTriageEnv:
    """Real-world email triage simulation environment with OpenEnv-like API."""

    MAX_STEPS_DEFAULT = 30

    def __init__(self, task_mode: TaskMode = "hard", seed: int = 42, max_steps: int = MAX_STEPS_DEFAULT):
        self.task_mode = task_mode
        self.seed = seed
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        self._emails: List[Email] = []
        self._gold: Dict[str, EmailGold] = {}
        self._inbox: List[InboxItemState] = []
        self._current_index = 0
        self._step_count = 0
        self._done = False
        self._previous_action_result = "Environment not started"
        self._action_history: List[Dict[str, object]] = []

        self._init_dataset()

    def _init_dataset(self) -> None:
        base_emails = [
            Email(
                email_id="e1",
                sender="security@yourbank-alerts.com",
                subject="Urgent: Verify Account Password",
                content="Your account is suspended. Click this link to verify your password now.",
                urgency_level=5,
                spam_probability=0.94,
            ),
            Email(
                email_id="e2",
                sender="manager@company.com",
                subject="Board deck needed by 4 PM",
                content="Please send the updated board deck and budget notes before 4 PM today.",
                urgency_level=5,
                spam_probability=0.03,
            ),
            Email(
                email_id="e3",
                sender="hr@company.com",
                subject="Policy acknowledgment reminder",
                content="Please acknowledge the updated leave policy by end of week.",
                urgency_level=3,
                spam_probability=0.05,
            ),
            Email(
                email_id="e4",
                sender="offers@shopping-deals.net",
                subject="Limited offer: 90% off coupon",
                content="Claim your 90% discount coupon. Offer expires in 10 minutes.",
                urgency_level=2,
                spam_probability=0.88,
            ),
            Email(
                email_id="e5",
                sender="client@acme.com",
                subject="Need project status update",
                content="Could you share milestone progress and blockers for this sprint?",
                urgency_level=4,
                spam_probability=0.04,
            ),
        ]

        self._gold = {
            "e1": EmailGold("spam", ("phishing", "security", "ignore"), 3),
            "e2": EmailGold("important", ("board", "deck", "send", "4 pm"), 1),
            "e3": EmailGold("normal", ("policy", "acknowledge", "week"), 2),
            "e4": EmailGold("spam", ("spam", "unsubscribe", "ignore"), 3),
            "e5": EmailGold("important", ("status", "milestone", "blockers"), 1),
        }

        self._emails = base_emails[:]
        self._rng.shuffle(self._emails)

    def _make_observation(self) -> Observation:
        current_email = None if self._done or self._current_index >= len(self._inbox) else self._inbox[self._current_index].email
        return Observation(
            current_email=current_email,
            inbox_state=[item.model_copy(deep=True) for item in self._inbox],
            previous_action_result=self._previous_action_result,
            task_mode=self.task_mode,
            step_count=self._step_count,
        )

    def _required_fields_done(self, item: InboxItemState) -> bool:
        if self.task_mode == "easy":
            return item.classified
        if self.task_mode == "medium":
            return item.replied
        return item.classified and item.replied and item.prioritized

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _semantic_similarity(reply_text: str, expected_keywords: Tuple[str, ...]) -> float:
        if not reply_text.strip():
            return 0.0
        lower_reply = reply_text.lower()
        hit_count = sum(1 for kw in expected_keywords if kw in lower_reply)
        if not expected_keywords:
            return 0.0
        return hit_count / len(expected_keywords)

    def reset(self) -> Observation:
        self._init_dataset()
        self._inbox = [InboxItemState(email=e) for e in self._emails]
        self._current_index = 0
        self._step_count = 0
        self._done = False
        self._previous_action_result = "Environment reset"
        self._action_history = []
        return self._make_observation()

    def state(self) -> Dict[str, object]:
        return {
            "task_mode": self.task_mode,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "step_count": self._step_count,
            "done": self._done,
            "current_index": self._current_index,
            "inbox": [item.model_dump() for item in self._inbox],
            "action_history": self._action_history[:],
        }

    def _advance_cursor(self) -> None:
        while self._current_index < len(self._inbox) and self._required_fields_done(self._inbox[self._current_index]):
            self._current_index += 1
        if self._current_index >= len(self._inbox):
            self._done = True

    def step(self, action: Action) -> StepResult:
        if self._done:
            observation = self._make_observation()
            reward = Reward(value=0.0, components={"done": 0.0})
            return StepResult(
                observation=observation,
                reward=reward,
                done=True,
                info={"message": "Episode already finished"},
            )

        self._step_count += 1
        current_item = self._inbox[self._current_index]
        gold = self._gold[current_item.email.email_id]

        reward_components = {
            "classification": 0.0,
            "reply": 0.0,
            "prioritization": 0.0,
            "wrong_action": 0.0,
            "unnecessary_step": 0.0,
        }

        message_parts: List[str] = []
        valid_action = True

        if action.action_type == "classify_email":
            if action.classification is None:
                valid_action = False
                reward_components["wrong_action"] -= 0.2
                message_parts.append("Missing classification label")
            elif current_item.classified:
                reward_components["unnecessary_step"] -= 0.1
                message_parts.append("Email already classified")
            else:
                current_item.classified = True
                current_item.classification = action.classification
                if action.classification == gold.classification:
                    reward_components["classification"] += 0.5
                    message_parts.append("Correct classification")
                else:
                    reward_components["wrong_action"] -= 0.2
                    message_parts.append("Wrong classification")

        elif action.action_type == "reply_email":
            if action.reply_text is None:
                valid_action = False
                reward_components["wrong_action"] -= 0.2
                message_parts.append("Missing reply text")
            elif current_item.replied:
                reward_components["unnecessary_step"] -= 0.1
                message_parts.append("Email already replied")
            else:
                current_item.replied = True
                current_item.reply_text = action.reply_text
                similarity = self._semantic_similarity(action.reply_text, gold.reply_keywords)
                reward_components["reply"] += round(0.3 * similarity, 4)
                if similarity < 0.34:
                    reward_components["wrong_action"] -= 0.2
                    message_parts.append("Low quality reply")
                else:
                    message_parts.append("Good quality reply")

        elif action.action_type == "prioritize_email":
            if action.priority_rank is None:
                valid_action = False
                reward_components["wrong_action"] -= 0.2
                message_parts.append("Missing priority rank")
            elif current_item.prioritized:
                reward_components["unnecessary_step"] -= 0.1
                message_parts.append("Email already prioritized")
            else:
                current_item.prioritized = True
                current_item.priority_rank = action.priority_rank
                if action.priority_rank == gold.priority_rank:
                    reward_components["prioritization"] += 0.2
                    message_parts.append("Correct prioritization")
                else:
                    reward_components["wrong_action"] -= 0.2
                    message_parts.append("Wrong prioritization")
        else:
            valid_action = False
            reward_components["wrong_action"] -= 0.2
            message_parts.append("Unsupported action")

        # Penalty for actions not relevant to task mode.
        if self.task_mode == "easy" and action.action_type != "classify_email":
            reward_components["unnecessary_step"] -= 0.1
            message_parts.append("Non-essential action in easy task")
        if self.task_mode == "medium" and action.action_type != "reply_email":
            reward_components["unnecessary_step"] -= 0.1
            message_parts.append("Non-essential action in medium task")

        raw_reward = sum(reward_components.values())
        step_reward = self._clamp01(raw_reward)

        self._advance_cursor()

        if self._step_count >= self.max_steps:
            self._done = True
            message_parts.append("Max steps reached")

        self._previous_action_result = "; ".join(message_parts) if message_parts else "Action processed"
        self._action_history.append(
            {
                "step": self._step_count,
                "action": action.model_dump(),
                "valid_action": valid_action,
                "reward_components": reward_components,
                "raw_reward": round(raw_reward, 4),
                "reward": round(step_reward, 4),
                "done": self._done,
                "result": self._previous_action_result,
            }
        )

        observation = self._make_observation()
        reward = Reward(value=step_reward, components=reward_components)
        info = {
            "raw_reward": round(raw_reward, 4),
            "task_progress": self._current_index / max(1, len(self._inbox)),
            "history_length": len(self._action_history),
        }
        return StepResult(observation=observation, reward=reward, done=self._done, info=info)

    def close(self) -> None:
        self._done = True

    @property
    def action_history(self) -> List[Dict[str, object]]:
        return self._action_history[:]

    @property
    def inbox_count(self) -> int:
        return len(self._inbox)
