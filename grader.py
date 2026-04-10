from __future__ import annotations


def grade_total_reward(total_reward: float) -> float:
    """
    Normalize reward into safe (0,1) range for OpenEnv validation
    """
    try:
        normalized = total_reward / (abs(total_reward) + 1)
        score = (normalized + 1) / 2
        score = max(0.01, min(0.99, score))
        return float(score)
    except Exception:
        return 0.5


if __name__ == "__main__":
    example_reward = 0.82
    print(grade_total_reward(example_reward))
