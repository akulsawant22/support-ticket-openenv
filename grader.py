from __future__ import annotations


def grade_total_reward(total_reward: float) -> float:
    """Deterministically convert cumulative reward into a normalized OpenEnv score."""
    score = round(total_reward, 4)
    score = max(0.01, min(0.99, score))
    return float(score)


if __name__ == "__main__":
    example_reward = 0.82
    print(grade_total_reward(example_reward))
