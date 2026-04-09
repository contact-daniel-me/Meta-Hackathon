"""
Grader module for EV Charging Environment.

This module provides the main grader interface used by OpenEnv
to evaluate agent performance on the EV charging tasks.
"""

import json
import sys
import os
import traceback
from typing import Dict, Any, Optional, List

from models import EnvironmentState, TaskConfig, Action
from tasks import get_task_config, grade_task
from environment import EVChargingEnvironment


def _clamp_score(value: float) -> float:
    """Clamp a score or reward to strictly (0.001, 0.999) - never exactly 0 or 1."""
    try:
        val = float(value)
        return max(0.001, min(0.999, val))
    except (ValueError, TypeError):
        return 0.001

def _deep_clamp(obj: Any) -> Any:
    """Recursively clamp all floats in a nested structure to satisfy OpenEnv requirement."""
    if isinstance(obj, dict):
        return {k: _deep_clamp(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_clamp(v) for v in obj]
    elif isinstance(obj, float):
        if obj == 0.0: return 0.001
        if obj == 1.0: return 0.999
        if -1.1 < obj < 1.1:
            return max(0.001, min(0.999, obj))
    return obj


class EVChargingGrader:
    """
    Main grader class for EV Charging Environment.
    """
    
    def __init__(self):
        """Initialize the grader."""
        self.environment: Optional[EVChargingEnvironment] = None
        self.current_difficulty: Optional[str] = None
    
    def setup(self, difficulty: str, seed: int = 42) -> Dict[str, Any]:
        """Set up the grader."""
        self.current_difficulty = difficulty
        task_config = get_task_config(difficulty)
        self.environment = EVChargingEnvironment(task_config, seed=seed)
        initial_observation = self.environment.reset()
        
        return {
            "difficulty": difficulty,
            "task_config": task_config.dict() if hasattr(task_config, "dict") else task_config,
            "initial_observation": initial_observation.dict() if hasattr(initial_observation, "dict") else str(initial_observation),
            "seed": seed
        }
    
    def evaluate_step(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single step."""
        if not self.environment:
            raise RuntimeError("Grader not set up.")
        
        action = Action(
            type=action_data.get("type", "wait"),
            station_id=action_data.get("station_id"),
            wait_time_minutes=action_data.get("wait_time_minutes", 0)
        )
        observation, reward, done, info = self.environment.step(action)
        
        reward_val = getattr(reward, 'value', reward) if hasattr(reward, 'value') else float(reward)
        clamped_reward_val = _clamp_score(reward_val)
        
        reward_dict = reward.dict() if hasattr(reward, "dict") else {"value": clamped_reward_val}
        reward_dict["value"] = clamped_reward_val
        if "breakdown" in reward_dict:
            reward_dict["breakdown"] = {
                k: _clamp_score(v) for k, v in reward_dict["breakdown"].items()
            }
            
        return _deep_clamp({
            "observation": observation.dict() if hasattr(observation, "dict") else str(observation),
            "reward": reward_dict,
            "done": done,
            "info": info,
            "step_count": self.environment.current_step
        })
    
    def evaluate_episode(self, action_sequence: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
        """Evaluate a complete episode."""
        if not self.current_difficulty:
            raise RuntimeError("Grader not set up.")
        
        self.environment.reset()
        step_results = []
        total_reward = 0.0
        done = False
        
        for action_data in action_sequence:
            if done: break
            step_result = self.evaluate_step(action_data)
            step_results.append(step_result)
            total_reward += step_result["reward"]["value"]
            done = step_result["done"]
        
        final_state = self.environment.state()
        grade = _clamp_score(grade_task(final_state))
        final_score = _clamp_score(getattr(final_state, 'score', 0.001))
        
        final_state_dict = final_state.dict() if hasattr(final_state, "dict") else {}
        final_state_dict["score"] = final_score
        
        return _deep_clamp({
            "grade": grade,
            "final_score": final_score,
            "total_reward": _clamp_score(total_reward),
            "steps_taken": len(step_results),
            "episode_done": done,
            "step_results": step_results,
            "final_state": final_state_dict
        })

    def get_grade(self, environment_state: EnvironmentState) -> float:
        """Get grade for state."""
        return _clamp_score(grade_task(environment_state))


def validate_submission(submission_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate submission payload."""
    results = {"errors": [], "warnings": []}
    if "actions" not in submission_data:
        results["errors"].append("Missing 'actions'")
    elif not isinstance(submission_data["actions"], list):
        results["errors"].append("'actions' must be list")
    if "difficulty" not in submission_data:
        results["errors"].append("Missing 'difficulty'")
    return results


def main():
    """Main entry point for CLI grading."""
    if len(sys.argv) < 2:
        print("Usage: python grader.py <submission_file.json> OR python grader.py <difficulty> <file>")
        sys.exit(1)
        
    if len(sys.argv) == 2:
        submission_file = sys.argv[1]
        difficulty = "all"
    else:
        difficulty = sys.argv[1]
        submission_file = sys.argv[2]
        
    try:
        if not os.path.exists(submission_file):
            print(f"Error: {submission_file} not found")
            sys.exit(1)
            
        with open(submission_file, 'r') as f:
            submission_data = json.load(f)
            
        if isinstance(submission_data, list):
            tasks = submission_data
        elif isinstance(submission_data, dict):
            if "difficulty" in submission_data:
                tasks = [submission_data]
            else:
                tasks = []
                for v in submission_data.values():
                    if isinstance(v, dict) and "difficulty" in v:
                        tasks.append(v)
        else:
            print("Error: Invalid format")
            sys.exit(1)

        if difficulty != "all":
            tasks = [t for t in tasks if t.get("difficulty") == difficulty]

        for i, task_data in enumerate(tasks):
            diff = task_data.get("difficulty", "easy")
            print(f"Grading Task {i+1}: {diff}")
            grader = EVChargingGrader()
            grader.setup(diff, seed=task_data.get("seed", 42))
            results = grader.evaluate_episode(task_data.get("actions", []))
            print(f"  Grade: {results['grade']:.4f}")

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()