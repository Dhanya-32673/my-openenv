from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Email(BaseModel):
    email_id: str
    sender: str
    subject: str
    content: str
    urgency_level: int = Field(ge=1, le=5)
    spam_probability: float = Field(ge=0.0, le=1.0)


class Action(BaseModel):
    action_type: Literal["classify_email", "reply_email", "prioritize_email"]
    classification: Optional[Literal["spam", "important", "normal"]] = None
    reply_text: Optional[str] = None
    priority_rank: Optional[int] = Field(default=None, ge=1, le=3)


class InboxItemState(BaseModel):
    email: Email
    classified: bool = False
    classification: Optional[str] = None
    replied: bool = False
    reply_text: Optional[str] = None
    prioritized: bool = False
    priority_rank: Optional[int] = None


class Observation(BaseModel):
    current_email: Optional[Email] = None
    inbox_state: List[InboxItemState]
    previous_action_result: str
    task_mode: Literal["easy", "medium", "hard"]
    step_count: int


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def round_reward(cls, v: float) -> float:
        return round(float(v), 4)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)
