from __future__ import annotations

import requests


def main() -> None:
    base_url = "http://127.0.0.1:8000"

    reset_response = requests.post(
        f"{base_url}/reset",
        json={"task_name": "easy", "seed": 0},
        timeout=10,
    )
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert isinstance(reset_payload.get("ticket"), str)
    assert isinstance(reset_payload.get("history"), list)
    assert isinstance(reset_payload.get("step_count"), int)

    step_response = requests.post(
        f"{base_url}/step",
        json={"action": "categorize_billing"},
        timeout=10,
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert isinstance(step_payload.get("observation"), dict)
    assert isinstance(step_payload.get("reward"), (int, float))
    assert isinstance(step_payload.get("done"), bool)
    assert isinstance(step_payload.get("info"), dict)

    state_response = requests.get(f"{base_url}/state", timeout=10)
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert isinstance(state_payload, dict)
    assert isinstance(state_payload.get("ticket"), str)
    assert isinstance(state_payload.get("history"), list)
    assert isinstance(state_payload.get("step_count"), int)


if __name__ == "__main__":
    main()
