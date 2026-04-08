from __future__ import annotations

from typing import Dict

from env.environment import SmartEmailTriageEnv


def grade_hard(env: SmartEmailTriageEnv) -> Dict[str, float]:
    """Deterministic grader combining correctness and efficiency for hard mode."""
    history = env.action_history
    if not history:
        return {
            "score": 0.0,
            "classification_accuracy": 0.0,
            "reply_quality": 0.0,
            "priority_accuracy": 0.0,
            "efficiency": 0.0,
        }

    class_actions = [h for h in history if h["action"]["action_type"] == "classify_email"]
    reply_actions = [h for h in history if h["action"]["action_type"] == "reply_email"]
    prior_actions = [h for h in history if h["action"]["action_type"] == "prioritize_email"]

    class_acc = (
        sum(1 for h in class_actions if h["reward_components"]["classification"] > 0) / len(class_actions)
        if class_actions
        else 0.0
    )

    reply_quality = (
        sum(max(0.0, min(1.0, float(h["reward_components"].get("reply", 0.0)) / 0.3)) for h in reply_actions)
        / len(reply_actions)
        if reply_actions
        else 0.0
    )

    prior_acc = (
        sum(1 for h in prior_actions if h["reward_components"]["prioritization"] > 0) / len(prior_actions)
        if prior_actions
        else 0.0
    )

    # Ideal hard flow is 3 actions per email.
    ideal_steps = max(1, env.inbox_count * 3)
    steps_taken = len(history)
    efficiency = max(0.0, min(1.0, ideal_steps / max(ideal_steps, steps_taken)))

    score = (0.35 * class_acc) + (0.30 * reply_quality) + (0.25 * prior_acc) + (0.10 * efficiency)
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "classification_accuracy": round(class_acc, 4),
        "reply_quality": round(reply_quality, 4),
        "priority_accuracy": round(prior_acc, 4),
        "efficiency": round(efficiency, 4),
    }
