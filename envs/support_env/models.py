from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ActionName = Literal[
    "categorize_billing",
    "categorize_technical",
    "categorize_general",
    "request_more_info",
    "refund_user",
    "escalate_to_human",
    "close_ticket",
]

TaskName = Literal["easy", "medium", "hard"]
TicketCategory = Literal["billing", "technical", "general"]


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ActionName = Field(description="Discrete support action selected by an agent.")
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional human-readable rationale for debugging or traceability.",
    )


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: str
    history: List[str]
    step_count: int


class State(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: TaskName
    ticket_id: str
    ticket: str
    category: TicketCategory
    expected_resolution: ActionName
    required_steps: List[ActionName]
    history: List[str]
    step_count: int
    max_steps: int
    resolved: bool
    done: bool
    total_reward: float
    last_action: Optional[ActionName] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: TaskName = "easy"
    seed: int = 0


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action

