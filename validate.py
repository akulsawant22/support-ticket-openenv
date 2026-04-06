from __future__ import annotations

from envs.support_env.models import Action
from envs.support_env.server.environment import SupportTicketEnvironment


def main() -> None:
    env = SupportTicketEnvironment()

    observation = env.reset(task_name="easy", seed=0)
    print("reset:", observation.model_dump())

    result = env.step(Action(name="categorize_billing", reasoning="validation check"))
    print("step:", result.model_dump())

    state = env.state()
    print("state:", state.model_dump() if state else None)


if __name__ == "__main__":
    main()
