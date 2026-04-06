from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from envs.support_env.models import ActionName, TaskName


@dataclass(frozen=True)
class TicketScenario:
    ticket_id: str
    ticket: str
    category: str
    expected_resolution: ActionName
    required_steps: List[ActionName]
    customer_tier: str
    needs_more_info: bool = False


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    name: TaskName
    description: str
    max_steps: int
    scenarios: List[TicketScenario]


TASKS: Dict[TaskName, TaskDefinition] = {
    "easy": TaskDefinition(
        id="support-easy",
        name="easy",
        description="Classification-only task where the agent must assign the correct ticket category.",
        max_steps=2,
        scenarios=[
            TicketScenario(
                ticket_id="E-001",
                ticket="I was charged twice for my monthly plan and need help fixing it.",
                category="billing",
                expected_resolution="refund_user",
                required_steps=["categorize_billing"],
                customer_tier="standard",
            ),
            TicketScenario(
                ticket_id="E-002",
                ticket="The app keeps crashing every time I upload a photo.",
                category="technical",
                expected_resolution="escalate_to_human",
                required_steps=["categorize_technical"],
                customer_tier="pro",
            ),
            TicketScenario(
                ticket_id="E-003",
                ticket="How do I reset my password from the mobile app?",
                category="general",
                expected_resolution="close_ticket",
                required_steps=["categorize_general"],
                customer_tier="standard",
            ),
        ],
    ),
    "medium": TaskDefinition(
        id="support-medium",
        name="medium",
        description="Classification plus one concrete resolution action.",
        max_steps=3,
        scenarios=[
            TicketScenario(
                ticket_id="M-001",
                ticket="I was charged twice after upgrading and want the duplicate payment reversed.",
                category="billing",
                expected_resolution="refund_user",
                required_steps=["categorize_billing", "refund_user"],
                customer_tier="business",
            ),
            TicketScenario(
                ticket_id="M-002",
                ticket="The desktop app crashes on launch after the latest update.",
                category="technical",
                expected_resolution="escalate_to_human",
                required_steps=["categorize_technical", "escalate_to_human"],
                customer_tier="pro",
            ),
            TicketScenario(
                ticket_id="M-003",
                ticket="How do I reset my password? I just need the right steps.",
                category="general",
                expected_resolution="close_ticket",
                required_steps=["categorize_general", "close_ticket"],
                customer_tier="standard",
            ),
        ],
    ),
    "hard": TaskDefinition(
        id="support-hard",
        name="hard",
        description="Full multi-step resolution including clarification when needed.",
        max_steps=5,
        scenarios=[
            TicketScenario(
                ticket_id="H-001",
                ticket="I think I was charged twice but I am not sure which invoice it came from.",
                category="billing",
                expected_resolution="refund_user",
                required_steps=["categorize_billing", "request_more_info", "refund_user"],
                customer_tier="business",
                needs_more_info=True,
            ),
            TicketScenario(
                ticket_id="H-002",
                ticket="The app crashes during checkout and support articles did not help.",
                category="technical",
                expected_resolution="escalate_to_human",
                required_steps=["categorize_technical", "request_more_info", "escalate_to_human"],
                customer_tier="enterprise",
                needs_more_info=True,
            ),
            TicketScenario(
                ticket_id="H-003",
                ticket="How do I reset my password if I no longer have access to my old email?",
                category="general",
                expected_resolution="close_ticket",
                required_steps=["categorize_general", "request_more_info", "close_ticket"],
                customer_tier="standard",
                needs_more_info=True,
            ),
        ],
    ),
}

