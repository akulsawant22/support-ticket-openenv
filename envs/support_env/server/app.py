from __future__ import annotations

import os

from fastapi import Body, FastAPI, HTTPException

from envs.support_env.models import AppConfig, Observation, ResetRequest, State, StepRequest, StepResult
from envs.support_env.server.environment import SupportEnvAPIWrapper


app = FastAPI()

app.state.config = AppConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("MODEL_NAME"),
    api_base_url=os.getenv("API_BASE_URL"),
    hf_token=os.getenv("HF_TOKEN"),
)
environment = SupportEnvAPIWrapper()


@app.post("/reset", response_model=Observation)
def reset(payload: dict | None = Body(default=None)) -> Observation:
    request = ResetRequest.model_validate(payload or {})
    return environment.reset(task_name=request.task_name, seed=request.seed)


@app.post("/step", response_model=StepResult)
def step(action: dict = Body(...)) -> StepResult:
    try:
        request = StepRequest.model_validate(action)
        return environment.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state", response_model=State | None)
def state() -> State | None:
    return environment.state()

@app.get("/")
def root():
    return {"message": "API is running. Use http://127.0.0.1:8000/docs"}