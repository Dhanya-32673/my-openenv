from __future__ import annotations

from env.environment import SmartEmailTriageEnv

TASK_NAME = "hard_full_inbox_management"
TASK_DESCRIPTION = "Complete classification, prioritization, and response for the full inbox."


def create_hard_env(seed: int = 42) -> SmartEmailTriageEnv:
    return SmartEmailTriageEnv(task_mode="hard", seed=seed, max_steps=30)
