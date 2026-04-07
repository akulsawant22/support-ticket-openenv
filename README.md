---
title: Customer Support Ticket Resolution Environment
emoji: "🧩"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: OpenEnv-compatible customer support ticket environment
---

# Customer Support Ticket Resolution Environment

Production-ready OpenEnv project for a hackathon-quality reinforcement learning
benchmark that simulates AI agents handling customer support tickets.

## Problem

Support automation is more than intent classification. Real systems must:

- detect the right category
- choose a safe next action
- gather missing details when needed
- resolve or escalate the ticket efficiently

This repository provides a deterministic environment for training or evaluating
agents on those behaviors with an OpenEnv-style API and a deployable FastAPI
server.

## Project Structure

```text
project/
├── envs/support_env/
│   ├── models.py
│   ├── client.py
│   ├── README.md
│   └── server/
│       ├── environment.py
│       ├── app.py
│       └── Dockerfile
├── inference.py
├── grader.py
├── tasks.py
├── openenv.yaml
├── requirements.txt
└── README.md
```

## Environment Overview

The environment exposes:

- `reset(task_name, seed)`
- `step(action)`
- `state()`

The `step()` method returns:

- `observation`
- `reward`
- `done`
- `info`

### Observation Space

```json
{
  "ticket": "string",
  "history": ["list", "of", "events"],
  "step_count": 0
}
```

### Action Space

- `categorize_billing`
- `categorize_technical`
- `categorize_general`
- `request_more_info`
- `refund_user`
- `escalate_to_human`
- `close_ticket`

### Tasks

- `easy`: classification only
- `medium`: classification + action
- `hard`: full multi-step resolution

### Reward Design

- correct classification: `+0.3`
- correct resolution: `+0.7`
- wrong action: `-0.2`
- inefficient extra step: `-0.1`
- final reward is capped between `0.0` and `1.0`

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

```bash
uvicorn envs.support_env.server.app:app --host 0.0.0.0 --port 8000
```

### 3. Call the environment

#### Reset

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d "{\"task_name\":\"easy\",\"seed\":0}"
```

#### Step

```bash
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d "{\"action\":{\"name\":\"categorize_billing\"}}"
```

#### State

```bash
curl http://127.0.0.1:8000/state
```

## Run Inference

The baseline agent is deterministic and reproducible.

```bash
python inference.py
```

It prints:

- `[START]` at task start
- `[STEP]` for each action
- `[END]` with task score and overall score

## Docker

Build and run:

```bash
docker build -f envs/support_env/server/Dockerfile -t support-env .
docker run -p 8000:8000 support-env
```

## Hugging Face Spaces

The project is container-friendly and suitable for Docker-based deployment on
Hugging Face Spaces:

- FastAPI app entrypoint included
- deterministic runtime
- lightweight dependency set
- no external services required

## Validation Notes

- Modular code organization
- Typed request and response models with Pydantic
- Deterministic tasks and grader
- Reproducible baseline inference
- Runtime far below 20 minutes
