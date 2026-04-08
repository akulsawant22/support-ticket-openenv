from __future__ import annotations

import os
from typing import Any, Dict

from openai import OpenAI

from envs.support_env.models import Action, ActionName
from envs.support_env.server.environment import SupportTicketEnvironment
from grader import grade_total_reward
from tasks import TASKS

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
    )
model = os.getenv("MODEL_NAME", "gpt-4o-mini")

ALLOWED_ACTIONS: tuple[ActionName, ...] = (
    "categorize_billing",
    "categorize_technical",
    "categorize_general",
    "request_more_info",
    "refund_user",
    "escalate_to_human",
    "close_ticket",
)


def get_action(observation: Dict[str, Any]) -> ActionName:
    if client is None:
        return "categorize_billing"

    messages = [
        {
            "role": "system",
            "content": "You are an expert customer support AI agent. Always choose the BEST next action.",
        },
        {
            "role": "user",
            "content": (
                f"ticket text: {observation['ticket']}\n"
                f"step count: {observation['step_count']}\n"
                f"allowed actions: {', '.join(ALLOWED_ACTIONS)}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        action = response.choices[0].message.content.strip()
    except Exception:
        return "categorize_billing"

    normalized_action = action.strip()
    if normalized_action not in ALLOWED_ACTIONS:
        return "categorize_billing"
    return normalized_action


def run_task(task_name: str, seed: int = 0) -> float:
    env = SupportTicketEnvironment()
    observation = env.reset(task_name=task_name, seed=seed)
    print("[START]")

    done = False

    while not done:
        action_name = get_action(observation.model_dump())
        result = env.step(Action(name=action_name, reasoning="openai agent"))
        observation = result.observation
        print(f"[STEP] {action_name} {result.reward}")
        done = result.done

    final_state = env.state()
    if final_state is None:
        raise RuntimeError("Environment state missing after run.")
    score = grade_total_reward(final_state.total_reward)
    print(f"[END] {score}")
    return score


def main() -> None:
    for index, task_name in enumerate(TASKS):
        run_task(task_name=task_name, seed=index)


if __name__ == "__main__":
    main()
