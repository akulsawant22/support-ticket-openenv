from __future__ import annotations


def grade_total_reward(total_reward: float) -> float:
    try:
        score = float(total_reward)

        if score != score:  # NaN check
            return 0.5

        if score <= 0:
            return 0.01
        if score >= 1:
            return 0.99

        return max(0.01, min(0.99, score))
    except Exception:
        return 0.5


if __name__ == "__main__":
    example_reward = 0.82
    print(grade_total_reward(example_reward))
