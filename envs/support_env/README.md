# Support Environment Package

This package contains the production-ready OpenEnv implementation for the
Customer Support Ticket Resolution Environment.

## Contents

- `models.py`: Shared Pydantic schemas for actions, observations, state, and API payloads.
- `client.py`: Lightweight HTTP client for interacting with the FastAPI environment server.
- `server/environment.py`: Deterministic customer support simulation and OpenEnv-compatible methods.
- `server/app.py`: FastAPI application exposing `/reset`, `/step`, and `/state`.

## OpenEnv Contract

The environment implements:

- `reset(task_name, seed)`
- `step(action)`
- `state()`

`step()` returns:

- `observation`
- `reward`
- `done`
- `info`

## Hugging Face Spaces

The `server/` directory includes a Dockerfile suitable for containerized
deployment on Hugging Face Spaces with FastAPI + Uvicorn.

