from __future__ import annotations

from env.environment import SmartEmailTriageEnv

TASK_NAME = "medium_reply_generation"
TASK_DESCRIPTION = "Generate context-appropriate replies for each email."


def create_medium_env(seed: int = 42) -> SmartEmailTriageEnv:
    return SmartEmailTriageEnv(task_mode="medium", seed=seed, max_steps=25)
