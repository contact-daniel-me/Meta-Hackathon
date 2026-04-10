"""
Inference script for EV Charging Environment.
Outputs structured [START]/[STEP]/[END] blocks to stdout as required by OpenEnv.
Uses the platform-provided LLM proxy (API_BASE_URL + API_KEY) for agent decisions.
"""

import os
import json
import sys
from typing import Dict, Any, Optional

import openai

from models import Observation, Action, ActionType
from environment import EVChargingEnvironment
from tasks import get_task_config, grade_task


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


def _get_openai_client() -> openai.OpenAI:
    """
    Create OpenAI client using ONLY the platform-injected environment variables.
    The hackathon platform injects API_BASE_URL and API_KEY for their LiteLLM proxy.
    We MUST use these exclusively — no fallbacks to personal credentials.
    """
    base_url = os.environ.get("API_BASE_URL", "")
    api_key = os.environ.get("API_KEY", "")

    # Log to stderr for debugging (stdout is reserved for structured output)
    print(f"[DEBUG] API_BASE_URL = {base_url!r}", file=sys.stderr, flush=True)
    print(f"[DEBUG] API_KEY present = {bool(api_key)}", file=sys.stderr, flush=True)

    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable not found. "
            "The platform must inject API_KEY and API_BASE_URL."
        )
    if not base_url:
        raise RuntimeError(
            "API_BASE_URL environment variable not found. "
            "The platform must inject API_BASE_URL for the LiteLLM proxy."
        )

    return openai.OpenAI(api_key=api_key, base_url=base_url)


class EVChargingAgent:
    """LLM-powered agent that makes decisions through the platform's API proxy."""

    def __init__(self):
        self.client = _get_openai_client()

    def decide_action(self, observation: Observation) -> Action:
        """Use OpenAI API to decide the next action based on the observation."""
        try:
            # Build a concise description of the current state
            stations_info = []
            for s in observation.stations:
                stations_info.append(
                    f"  - {s.id}: status={s.status.value}, "
                    f"lat={float(s.latitude):.4f}, lon={float(s.longitude):.4f}, "
                    f"price=${float(s.price_per_kwh):.2f}/kWh, "
                    f"queue={s.current_queue_length}/{s.max_queue_length}"
                )
            stations_text = "\n".join(stations_info) if stations_info else "  (none)"

            prompt = f"""You are an EV charging station optimizer. Pick the best action.

Current EV state:
- Battery: {observation.ev.current_battery_percent}%
- Location: ({float(observation.current_location_lat):.4f}, {float(observation.current_location_lon):.4f})
- Budget remaining: ${float(observation.budget_remaining):.2f}
- Time remaining: {float(observation.time_remaining_hours):.2f} hours

Available stations:
{stations_text}

Actions available:
1. select_station <station_id> - Go to a station and start charging
2. wait <minutes> - Wait at current location (5-30 min)
3. move_to_next_station - Move without selecting

Respond with ONLY a JSON object, no other text:
{{"action": "select_station", "station_id": "<id>"}}
or {{"action": "wait", "minutes": 10}}
or {{"action": "move_to_next_station"}}
"""
        except Exception as e:
            print(f"[WARN] Prompt build failed: {e}", file=sys.stderr, flush=True)
            return self._heuristic_action(observation)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an EV charging optimization agent. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            content = response.choices[0].message.content.strip()
            # Parse the JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            decision = json.loads(content)
            action_type = decision.get("action", "wait")

            if action_type == "select_station" and decision.get("station_id"):
                # Verify the station_id exists
                valid_ids = {s.id for s in observation.stations}
                sid = decision["station_id"]
                if sid in valid_ids:
                    return Action(type=ActionType.SELECT_STATION, station_id=sid)
                # If invalid ID, fall through to heuristic

            elif action_type == "wait":
                minutes = int(decision.get("minutes", 10))
                minutes = max(5, min(30, minutes))
                return Action(type=ActionType.WAIT, wait_time_minutes=minutes)

            elif action_type == "move_to_next_station":
                return Action(type=ActionType.MOVE_TO_NEXT_STATION)

        except Exception as e:
            # Log the error but don't crash — fall through to heuristic
            print(f"[WARN] LLM call failed, using heuristic: {e}", file=sys.stderr, flush=True)

        # Heuristic fallback: pick nearest available station
        return self._heuristic_action(observation)

    def _heuristic_action(self, observation: Observation) -> Action:
        """Fallback heuristic: pick the nearest available station."""
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
