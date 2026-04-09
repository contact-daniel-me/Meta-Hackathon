"""
Inference script for EV Charging Environment.
Outputs structured [START]/[STEP]/[END] blocks to stdout as required by OpenEnv.
"""

import os
import json
import sys
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import openai

from models import Observation, Action, ActionType
from environment import EVChargingEnvironment
from tasks import get_task_config, grade_task

load_dotenv()


def _clamp(val):
    """Force any numeric value into strictly (0.001, 0.999)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (float, int)):
        return max(0.001, min(0.999, round(float(val), 4)))
    return val


def _deep_clamp(obj):
    """Recursively clamp every number in a nested structure."""
    if isinstance(obj, dict):
        return {k: _deep_clamp(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_clamp(v) for v in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (float, int)):
        return max(0.001, min(0.999, round(float(obj), 4)))
    return obj


class EVChargingAgent:
    """Simple heuristic agent for reliable execution."""

    def decide_action(self, observation: Observation) -> Action:
        available = [s for s in observation.stations if s.status == "available"]
        if available:
            best = min(
                available,
                key=lambda s: abs(s.latitude - observation.current_location_lat)
                + abs(s.longitude - observation.current_location_lon),
            )
            return Action(type=ActionType.SELECT_STATION, station_id=best.id)
        return Action(type=ActionType.WAIT, wait_time_minutes=10)


def run_task(difficulty: str, seed: int = 42) -> Dict[str, Any]:
    """Run one task and print structured [START]/[STEP]/[END] to stdout."""
    env = EVChargingEnvironment(get_task_config(difficulty), seed=seed)
    agent = EVChargingAgent()
    obs = env.reset()

    # --- [START] ---
    print(f"[START] task={difficulty}", flush=True)

    total_reward = 0.0
    step_num = 0

    for _ in range(env.task_config.max_steps):
        action = agent.decide_action(obs)
        obs, reward, done, info = env.step(action)
        step_num += 1
        r = _clamp(reward.value)
        total_reward += r

        # --- [STEP] ---
        print(f"[STEP] step={step_num} reward={r}", flush=True)

        if done:
            break

    state = env.state()
    score = _clamp(grade_task(state))

    # --- [END] ---
    print(f"[END] task={difficulty} score={score} steps={step_num}", flush=True)

    return {
        "difficulty": difficulty,
        "seed": seed,
        "grade": score,
        "total_reward": _clamp(total_reward),
        "final_score": _clamp(state.score),
        "steps_taken": _clamp(step_num),
        "done": True,
    }


def main():
    output_file = "submission.json"
    all_results = []
    for diff in ["easy", "medium", "hard"]:
        result = run_task(diff)
        all_results.append(_deep_clamp(result))

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
