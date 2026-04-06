from __future__ import annotations

from fastapi import FastAPI, HTTPException

from envs.support_env.models import Observation, ResetRequest, State, StepRequest, StepResult
from envs.support_env.server.environment import SupportTicketEnvironment


app = FastAPI(
    title="Customer Support Ticket Resolution Environment",
    description="OpenEnv-compatible support ticket simulation for reinforcement learning agents.",
    version="1.0.0",
)

environment = SupportTicketEnvironment()


@app.post("/reset", response_model=Observation)
def reset_environment(request: ResetRequest) -> Observation:
    return environment.reset(task_name=request.task_name, seed=request.seed)


@app.post("/step", response_model=StepResult)
def step_environment(request: StepRequest) -> StepResult:
    try:
        return environment.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state", response_model=State | None)
def get_state() -> State | None:
    return environment.state()

