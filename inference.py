from __future__ import annotations

from typing import Dict, List

from envs.support_env.models import Action, ActionName, Observation
from envs.support_env.server.environment import SupportTicketEnvironment
from grader import grade_total_reward
from tasks import TASKS


CATEGORY_RULES: Dict[str, ActionName] = {
    "charged twice": "categorize_billing",
    "invoice": "categorize_billing",
    "password": "categorize_general",
    "how do i": "categorize_general",
    "crash": "categorize_technical",
    "checkout": "categorize_technical",
}


RESOLUTION_RULES: Dict[str, ActionName] = {
    "billing": "refund_user",
    "technical": "escalate_to_human",
    "general": "close_ticket",
}


def choose_category_action(ticket: str) -> ActionName:
    normalized = ticket.lower()
    for keyword, action in CATEGORY_RULES.items():
        if keyword in normalized:
            return action
    return "categorize_general"


def choose_follow_up_action(observation: Observation, performed: List[ActionName]) -> ActionName:
    normalized = observation.ticket.lower()
    category_action = choose_category_action(observation.ticket)

    if category_action not in performed:
        return category_action

    inferred_category = {
        "categorize_billing": "billing",
        "categorize_technical": "technical",
        "categorize_general": "general",
    }[category_action]

    needs_more_info = any(phrase in normalized for phrase in ("not sure", "old email", "did not help"))
    if needs_more_info and "request_more_info" not in performed:
        return "request_more_info"

    return RESOLUTION_RULES[inferred_category]


def run_task(task_name: str, seed: int = 0) -> float:
    env = SupportTicketEnvironment()
    observation = env.reset(task_name=task_name, seed=seed)
    print(f"[START] task={task_name} seed={seed} ticket={observation.ticket}")

    done = False
    performed: List[ActionName] = []

    while not done:
        action_name = choose_follow_up_action(observation, performed)
        performed.append(action_name)
        result = env.step(Action(name=action_name, reasoning="rule-based baseline"))
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

