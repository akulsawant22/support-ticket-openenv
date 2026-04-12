from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from envs.support_env.models import Action, ActionName, Observation, State, StepResult, TaskName
from tasks import TASKS, TaskDefinition, TicketScenario


CATEGORY_TO_ACTION: Dict[str, ActionName] = {
    "billing": "categorize_billing",
    "technical": "categorize_technical",
    "general": "categorize_general",
}


RESOLUTION_HINTS: Dict[ActionName, str] = {
    "refund_user": "Refund approved and duplicate charge remediation started.",
    "escalate_to_human": "Ticket escalated to a human specialist for advanced handling.",
    "close_ticket": "Ticket closed after providing the requested resolution.",
    "request_more_info": "Additional customer details requested before resolution.",
    "categorize_billing": "Ticket classified as billing.",
    "categorize_technical": "Ticket classified as technical.",
    "categorize_general": "Ticket classified as general.",
}


def clamp_reward(value: float) -> float:
    return max(0.01, min(0.99, round(value, 4)))


@dataclass(frozen=True)
class RewardBreakdown:
    classification_reward: float = 0.0
    resolution_reward: float = 0.0
    wrong_action_penalty: float = 0.0
    inefficiency_penalty: float = 0.0

    @property
    def raw_total(self) -> float:
        return round(
            self.classification_reward
            + self.resolution_reward
            + self.wrong_action_penalty
            + self.inefficiency_penalty,
            4,
        )

    @property
    def total(self) -> float:
        raw = self.raw_total
        return clamp_reward(raw)


class SupportTicketEnvironment:
    """Deterministic RL-style environment for customer support ticket resolution."""

    def __init__(self) -> None:
        self._state: Optional[State] = None

    def reset(self, task_name: TaskName = "easy", seed: int = 0) -> Observation:
        task = TASKS[task_name]
        scenario = self._select_scenario(task, seed)
        self._state = State(
            task_id=task.id,
            task_name=task.name,
            ticket_id=scenario.ticket_id,
            ticket=scenario.ticket,
            category=scenario.category,
            expected_resolution=scenario.expected_resolution,
            required_steps=scenario.required_steps,
            history=[f"task={task.name}", f"ticket_opened:{scenario.ticket_id}"],
            step_count=0,
            max_steps=task.max_steps,
            resolved=False,
            done=False,
            total_reward=0.0,
            metadata={
                "difficulty": task.name,
                "customer_tier": scenario.customer_tier,
                "needs_more_info": scenario.needs_more_info,
            },
        )
        return self._observation()

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step() is called.")
        if self._state.done:
            return StepResult(
                observation=self._observation(),
                reward=0.01,
                done=True,
                info={"message": "Episode already finished."},
            )

        reward = RewardBreakdown()
        info: Dict[str, Any] = {
            "task_name": self._state.task_name,
            "ticket_id": self._state.ticket_id,
            "expected_category_action": CATEGORY_TO_ACTION[self._state.category],
            "expected_resolution_action": self._state.expected_resolution,
        }

        expected_next = self._next_expected_action()
        self._state.step_count += 1
        self._state.last_action = action.name
        self._state.history.append(f"agent:{action.name}")
        if action.name == expected_next:
            reward = self._reward_for_correct_action(action.name)
            info["status"] = "correct"
            if action.name in ("refund_user", "escalate_to_human", "close_ticket"):
                self._state.resolved = True
                self._state.history.append(f"system:{RESOLUTION_HINTS[action.name]}")
            elif action.name == "request_more_info":
                self._state.history.append(
                    "system:Customer provided the requested missing information."
                )
            else:
                self._state.history.append(f"system:{RESOLUTION_HINTS[action.name]}")
        else:
            reward = RewardBreakdown(wrong_action_penalty=-0.2)
            info["status"] = "incorrect"
            info["expected_next_action"] = expected_next
            self._state.history.append(
                f"system:Action '{action.name}' was not appropriate for the current state."
            )

        if self._state.step_count > len(self._state.required_steps):
            reward = RewardBreakdown(
                classification_reward=reward.classification_reward,
                resolution_reward=reward.resolution_reward,
                wrong_action_penalty=reward.wrong_action_penalty,
                inefficiency_penalty=-0.1,
            )
            info["inefficient"] = True

        if self._is_episode_complete():
            self._state.done = True
            info["episode_outcome"] = "resolved" if self._state.resolved else "incomplete"
        elif self._state.step_count >= self._state.max_steps:
            self._state.done = True
            self._state.history.append("system:Maximum step budget reached.")
            info["episode_outcome"] = "max_steps_exceeded"

        self._state.total_reward = clamp_reward(self._state.total_reward + reward.raw_total)
        info["total_reward"] = self._state.total_reward
        info["raw_reward_delta"] = reward.raw_total
        return StepResult(
            observation=self._observation(),
            reward=reward.total,
            done=self._state.done,
            info=info,
        )

    def state(self) -> Optional[State]:
        return self._state

    def _select_scenario(self, task: TaskDefinition, seed: int) -> TicketScenario:
        index = seed % len(task.scenarios)
        return task.scenarios[index]

    def _observation(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized.")
        return Observation(
            ticket=self._state.ticket,
            history=list(self._state.history),
            step_count=self._state.step_count,
        )

    def _next_expected_action(self) -> ActionName:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized.")

        completed_steps = [
            entry.removeprefix("agent:")
            for entry in self._state.history
            if entry.startswith("agent:")
        ]
        for required in self._state.required_steps:
            if required not in completed_steps:
                return required
        return self._state.required_steps[-1]

    def _reward_for_correct_action(self, action_name: ActionName) -> RewardBreakdown:
        classification_actions = {
            "categorize_billing",
            "categorize_technical",
            "categorize_general",
        }
        if action_name in classification_actions:
            return RewardBreakdown(classification_reward=0.3)
        if action_name in {"refund_user", "escalate_to_human", "close_ticket"}:
            return RewardBreakdown(resolution_reward=0.7)
        return RewardBreakdown()

    def _is_episode_complete(self) -> bool:
        if self._state is None:
            return False

        performed = [
            entry.removeprefix("agent:")
            for entry in self._state.history
            if entry.startswith("agent:")
        ]
        for step in self._state.required_steps:
            if step not in performed:
                return False
        return True


class SupportEnvAPIWrapper:
    """Thin API wrapper that preserves existing environment behavior."""

    def __init__(self, env: Optional[SupportTicketEnvironment] = None) -> None:
        self._env = env or SupportTicketEnvironment()

    def reset(self, task_name: TaskName = "easy", seed: int = 0) -> Observation:
        return self._env.reset(task_name=task_name, seed=seed)

    def step(self, action: Action | str) -> StepResult:
        resolved_action = action if isinstance(action, Action) else Action(name=action)
        return self._env.step(resolved_action)

    def state(self) -> Optional[State]:
        return self._env.state()
