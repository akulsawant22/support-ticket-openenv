from __future__ import annotations

from typing import Optional

import requests

from envs.support_env.models import Action, Observation, ResetRequest, State, StepResult


class SupportEnvClient:
    """Thin HTTP client for the customer support ticket environment."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, task_name: str = "easy", seed: int = 0) -> Observation:
        payload = ResetRequest(task_name=task_name, seed=seed).model_dump()
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return Observation.model_validate(response.json())

    def step(self, action: Action) -> StepResult:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return StepResult.model_validate(response.json())

    def state(self) -> Optional[State]:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload is None:
            return None
        return State.model_validate(payload)

