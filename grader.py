from __future__ import annotations


def safe_score(score: float) -> float:
    return float(max(0.01, min(0.99, score)))


def grade_total_reward(total_reward: float) -> float:
    try:
        normalized = total_reward / (abs(total_reward) + 1)
        score = (normalized + 1) / 2
    except Exception:
        score = 0.5

    if score <= 0:
        score = 0.01
    elif score >= 1:
        score = 0.99

    return safe_score(score)


if __name__ == "__main__":
    example_reward = 0.82
    print(grade_total_reward(example_reward))
