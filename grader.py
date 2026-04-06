from __future__ import annotations


def grade_total_reward(total_reward: float) -> float:
    """Deterministically convert cumulative reward into a normalized OpenEnv score."""
    return max(0.0, min(1.0, round(total_reward, 4)))


if __name__ == "__main__":
    example_reward = 0.82
    print(grade_total_reward(example_reward))

