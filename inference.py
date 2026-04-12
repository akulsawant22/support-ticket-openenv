from __future__ import annotations

import os
from typing import Any, Dict

from openai import OpenAI

from envs.support_env.models import Action, ActionName
from envs.support_env.server.environment import SupportTicketEnvironment
from tasks import TASKS

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"],
)

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
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a support ticket classifier."},
                {"role": "user", "content": observation["ticket"]},
            ],
        )
        text = response.choices[0].message.content.lower()
    except Exception:
        return "categorize_billing"

    if "billing" in text:
        action = "categorize_billing"
    elif "technical" in text or "crash" in text or "bug" in text:
        action = "categorize_technical"
    else:
        action = "categorize_general"

    if action not in ALLOWED_ACTIONS:
        return "categorize_billing"
    return action


def run_task(task_name: str, seed: int = 0) -> float:
    env = SupportTicketEnvironment()
    observation = env.reset(task_name=task_name, seed=seed)
    print("[START]")

    done = False

    while not done:
        action_name = get_action(observation.model_dump())
        result = env.step(Action(name=action_name, reasoning="openai agent"))

        if result.reward <= 0:
            result.reward = 0.01
        elif result.reward >= 1:
            result.reward = 0.99

        observation = result.observation
        print(f"[STEP] {action_name} {result.reward}")
        done = result.done

    final_state = env.state()
    if final_state is None:
        raise RuntimeError("Environment state missing after run.")
    total_reward = final_state.total_reward

    if total_reward <= 0:
        total_reward = 0.01
    elif total_reward >= 1:
        total_reward = 0.99

    final_score = float(total_reward)
    print("RAW REWARD:", final_state.total_reward)
    print("CLAMPED FINAL SCORE:", final_score)
    print(f"[END] {final_score}")
    return float(final_score)


def main() -> None:
    for index, task_name in enumerate(TASKS):
        run_task(task_name=task_name, seed=index)


if __name__ == "__main__":
    main()
