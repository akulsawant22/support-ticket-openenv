from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from envs.support_env.models import AppConfig, Observation, ResetRequest, State, StepRequest, StepResult
from envs.support_env.server.environment import SupportEnvAPIWrapper


app = FastAPI(
    title="Customer Support Ticket Resolution Environment",
    description="OpenEnv-compatible support ticket simulation for reinforcement learning agents.",
    version="1.0.0",
)

app.state.config = AppConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("MODEL_NAME"),
    api_base_url=os.getenv("API_BASE_URL"),
    hf_token=os.getenv("HF_TOKEN"),
)
environment = SupportEnvAPIWrapper()


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
