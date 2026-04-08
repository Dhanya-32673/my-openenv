from __future__ import annotations

from env.environment import SmartEmailTriageEnv

TASK_NAME = "easy_classification"
TASK_DESCRIPTION = "Classify each email as spam/important/normal with high accuracy."


def create_easy_env(seed: int = 42) -> SmartEmailTriageEnv:
    return SmartEmailTriageEnv(task_mode="easy", seed=seed, max_steps=20)
