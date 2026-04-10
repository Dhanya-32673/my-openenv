from __future__ import annotations

from typing import Dict

from env.environment import SmartEmailTriageEnv


def _strict_unit_interval(value: float) -> float:
    return max(0.0001, min(0.9999, value))


def grade_medium(env: SmartEmailTriageEnv) -> Dict[str, float]:
    """Deterministic grader for medium task using normalized reply quality."""
    history = env.action_history
    reply_actions = [h for h in history if h["action"]["action_type"] == "reply_email"]
    if not reply_actions:
        return {"score": 0.0001, "reply_quality": 0.0}

    # Reply component is already normalized to [0, 0.3], so divide by 0.3.
    normalized_quality = []
    for h in reply_actions:
        reply_component = float(h["reward_components"].get("reply", 0.0))
        normalized_quality.append(max(0.0, min(1.0, reply_component / 0.3)))

    avg_quality = sum(normalized_quality) / len(normalized_quality)
    score = _strict_unit_interval(avg_quality)
    return {"score": round(score, 4), "reply_quality": round(avg_quality, 4)}
