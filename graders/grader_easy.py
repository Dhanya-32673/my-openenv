from __future__ import annotations

from typing import Dict

from env.environment import SmartEmailTriageEnv


def _strict_unit_interval(value: float) -> float:
    return max(0.0001, min(0.9999, value))


def grade_easy(env: SmartEmailTriageEnv) -> Dict[str, float]:
    """Deterministic grader for easy task based on classification accuracy."""
    history = env.action_history
    classify_actions = [h for h in history if h["action"]["action_type"] == "classify_email"]
    if not classify_actions:
        return {"score": 0.0001, "accuracy": 0.0}

    correct = 0
    total = 0
    for h in classify_actions:
        total += 1
        if h["reward_components"]["classification"] > 0:
            correct += 1

    accuracy = correct / total if total else 0.0
    score = _strict_unit_interval(accuracy)
    return {"score": round(score, 4), "accuracy": round(accuracy, 4)}
