"""
Grader module for EV Charging Environment.

This module provides the main grader interface used by OpenEnv
to evaluate agent performance on the EV charging tasks.
"""

import json
import sys
import traceback
from typing import Dict, Any, Optional, List

from models import EnvironmentState, TaskConfig, Action
from tasks import get_task_config, grade_task
from environment import EVChargingEnvironment


# ---------------------------------------------------------------------------
# Score safety helper — enforces OpenEnv strict (0, 1) open-interval requirement
# ---------------------------------------------------------------------------
def _clamp_score(value: float) -> float:
    """Clamp a score or reward to strictly (0.001, 0.999) — never exactly 0 or 1."""
    try:
        val = float(value)
        return max(0.001, min(0.999, val))
    except (ValueError, TypeError):
        return 0.001


class EVChargingGrader:
    """
    Main grader class for EV Charging Environment.
    
    This class implements the grader interface expected by OpenEnv
    for evaluating agent submissions.
    """
    
    def __init__(self):
        """Initialize the grader."""
        self.environment: Optional[EVChargingEnvironment] = None
        self.current_difficulty: Optional[str] = None
    
    def setup(self, difficulty: str, seed: int = 42) -> Dict[str, Any]:
        """
        Set up the grader for a specific difficulty level.
        
        Args:
            difficulty: Task difficulty (easy, medium, hard)
            seed: Random seed for reproducibility
            
        Returns:
            Setup information
        """
        self.current_difficulty = difficulty
        
        # Get task configuration
        task_config = get_task_config(difficulty)
        
        # Create environment
        self.environment = EVChargingEnvironment(task_config, seed=seed)
        
        # Reset environment to get initial state
        initial_observation = self.environment.reset()
        
        return {
            "difficulty": difficulty,
            "task_config": task_config.dict() if hasattr(task_config, "dict") else task_config,
            "initial_observation": initial_observation.dict() if hasattr(initial_observation, "dict") else str(initial_observation),
            "seed": seed
        }
    
    def evaluate_step(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single step/action.
        
        Args:
            action_data: Action data from the agent
            
        Returns:
            Step evaluation results with clamped rewards
        """
        if not self.environment:
            raise RuntimeError("Grader not set up. Call setup() first.")
        
        # Convert action data to Action object safely
        action = Action(
            type=action_data.get("type", "wait"),
            station_id=action_data.get("station_id"),
            wait_time_minutes=action_data.get("wait_time_minutes", 0)
        )
        
        # Execute step
        observation, reward, done, info = self.environment.step(action)
        
        # Extract reward value and clamp it
        reward_val = getattr(reward, 'value', reward) if hasattr(reward, 'value') else float(reward)
        clamped_reward_val = _clamp_score(reward_val)
        
        # Update reward dict and recursively clamp breakdown if it exists
        reward_dict = reward.dict() if hasattr(reward, "dict") else {"value": clamped_reward_val}
        reward_dict["value"] = clamped_reward_val
        if "breakdown" in reward_dict:
            reward_dict["breakdown"] = {
                k: _clamp_score(v) for k, v in reward_dict["breakdown"].items()
            }
        return {
            "observation": observation.dict() if hasattr(observation, "dict") else str(observation),
            "reward": reward_dict,
            "done": done,
            "info": info,
            "step_count": self.environment.current_step
        }
    
    def evaluate_episode(self, action_sequence: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
        """
        Evaluate a complete episode with a sequence of actions.
        
        Args:
            action_sequence: List of action dictionaries
            seed: Random seed for reproducibility
            
        Returns:
            Episode evaluation results with all scores strictly in (0, 1)
        """
        if not self.current_difficulty:
            raise RuntimeError("Grader not set up. Call setup() first.")
        
        # Reset environment
        self.environment.reset()
        
        # Execute all actions
        step_results = []
        total_reward = 0.0
        done = False
        
        for action_data in action_sequence:
            if done:
                break
                
            step_result = self.evaluate_step(action_data)
            step_results.append(step_result)
            total_reward += step_result["reward"]["value"]
            done = step_result["done"]
        
        # Get final state and grade — strictly clamp to (0.001, 0.999)
        final_state = self.environment.state()
        grade = _clamp_score(grade_task(final_state))
        
        # Ensure final state score is also clamped
        raw_final_score = getattr(final_state, 'score', 0.001)
        final_score = _clamp_score(raw_final_score)
        
        # Prepare final state dict with clamped score
        final_state_dict = final_state.dict() if hasattr(final_state, "dict") else {}
        final_state_dict["score"] = final_score
        
        return {
            "grade": grade,
            "final_score": final_score,
            "total_reward": _clamp_score(total_reward),
            "steps_taken": len(step_results),
            "episode_done": done,
            "step_results": step_results,
            "final_state": final_state_dict
        }

    def get_grade(self, environment_state: EnvironmentState) -> float:
        """
        Get grade for a given environment state.
        
        Returns:
            Grade strictly between 0.001 and 0.999
        """
        return _clamp_score(grade_task(environment_state))


def validate_submission(submission_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate the incoming submission payload."""
    results = {"errors": [], "warnings": []}
    
    if "actions" not in submission_data:
        results["errors"].append("Missing 'actions' array in submission")
    elif not isinstance(submission_data["actions"], list):
        results["errors"].append("'actions' must be an array")
        
    if "difficulty" not in submission_data:
        results["errors"].append("Missing 'difficulty' in submission")
        
    return results


def main():
    """Main entry point for command-line grading."""
    if len(sys.argv) < 2:
        print("Usage: python grader.py <submission_file.json>")
        sys.exit(1)
        
    submission_file = sys.argv[1]
    
    try:
        with open(submission_file, 'r') as f:
            submission_data = json.load(f)
            
        # Support both single object and list of objects
        if isinstance(submission_data, list):
            tasks = submission_data
        else:
            tasks = [submission_data]
            
        for i, task_data in enumerate(tasks):
            difficulty = task_data.get("difficulty", "easy")
            print(f"\n--- Evaluating Task {i+1}: {difficulty} ---")
            
            validation = validate_submission(task_data)
            if validation["errors"]:
                print(f"Errors in task {i+1}:")
                for err in validation["errors"]:
                    print(f"  - {err}")
                continue

            grader = EVChargingGrader()
            grader.setup(difficulty, seed=task_data.get("seed", 42))
            
            results = grader.evaluate_episode(task_data["actions"])
            
            print(f"Results:")
            print(f"  Grade:        {results['grade']:.4f}")
            print(f"  Final Score:  {results['final_score']:.4f}")
            print(f"  Total Reward: {results['total_reward']:.4f}")
            print(f"  Steps Taken:  {results['steps_taken']}")
            
            # Save results
            suffix = f"_{difficulty}_results.json"
            output_file = submission_file.replace('.json', suffix)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {output_file}")
            
    except Exception as e:
        print(f"Error evaluating submission: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()