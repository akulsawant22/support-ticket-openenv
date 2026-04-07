from __future__ import annotations

import os
from typing import Any, Dict

from openai import OpenAI

from envs.support_env.models import Action, ActionName, Observation
from envs.support_env.server.environment import SupportTicketEnvironment
from grader import grade_total_reward
from tasks import TASKS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=messages,
            temperature=0,
        )
        action = response.choices[0].message.content.strip()
    except Exception:
        return "request_more_info"

    normalized_action = action.strip()
    if normalized_action not in ALLOWED_ACTIONS:
        return "request_more_info"
    return normalized_action


def run_task(task_name: str, seed: int = 0) -> float:
    env = SupportTicketEnvironment()
    observation = env.reset(task_name=task_name, seed=seed)
    print(f"[START] task={task_name} seed={seed} ticket={observation.ticket}")

    done = False

    while not done:
        action_name = get_action(observation.model_dump())
        result = env.step(Action(name=action_name, reasoning="openai agent"))
        observation = result.observation
        print(
            f"[STEP] task={task_name} step={observation.step_count} "
            f"action={action_name} reward={result.reward:.2f} done={result.done}"
        )
        done = result.done

    final_state = env.state()
    if final_state is None:
        raise RuntimeError("Environment state missing after run.")
    score = grade_total_reward(final_state.total_reward)
    print(f"[END] task={task_name} total_reward={final_state.total_reward:.2f} score={score:.2f}")
    return score


def main() -> None:
    scores = []
    for index, task_name in enumerate(TASKS):
        scores.append(run_task(task_name=task_name, seed=index))
    overall = round(sum(scores) / len(scores), 4)
    print(f"[END] overall_score={overall:.2f}")


if __name__ == "__main__":
    main()
