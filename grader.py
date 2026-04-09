"""
Grader module for EV Charging Environment.

This module provides the main grader interface used by OpenEnv
to evaluate agent performance on the EV charging tasks.
"""

import json
import sys
from typing import Dict, Any, Optional

from models import EnvironmentState, TaskConfig
from tasks import get_task_config, grade_task
from environment import EVChargingEnvironment


# ---------------------------------------------------------------------------
# Score safety helper — enforces OpenEnv strict (0, 1) open-interval requirement
# ---------------------------------------------------------------------------
def _clamp_score(value: float) -> float:
    """Clamp a score to strictly (0.001, 0.999) — never exactly 0 or 1."""
    return max(0.001, min(0.999, float(value)))


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
            "task_config": task_config.dict(),   # Pydantic v2: use .dict()
            "initial_observation": initial_observation.dict(),
            "seed": seed
        }

    def evaluate_step(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single step/action.

        Args:
            action_data: Action data from the agent

        Returns:
            Step evaluation results
        """
        if not self.environment:
            raise RuntimeError("Grader not set up. Call setup() first.")

        # Convert action data to Action object
        from models import Action, ActionType
        action = Action(**action_data)

        # Execute step
        observation, reward, done, info = self.environment.step(action)

        reward_dict = reward.dict()
        reward_dict["value"] = _clamp_score(reward_dict["value"])

        return {
            "observation": observation.dict(),
            "reward": reward_dict,
            "done": done,
            "info": info,
            "step_count": self.environment.current_step
        }

    def evaluate_episode(self, action_sequence: list, seed: int = 42) -> Dict[str, Any]:
        """
        Evaluate a complete episode with a sequence of actions.

        Args:
            action_sequence: List of action dictionaries
            seed: Random seed for reproducibility

        Returns:
            Episode evaluation results with all scores strictly in (0, 1)
        """
        if not self.current_difficulty:
            raise RuntimeError("Difficulty not set. Call setup() first.")

        # Set up environment
        setup_info = self.setup(self.current_difficulty, seed)

        # Execute all actions
        step_results = []
        total_reward = 0.0

        for action_data in action_sequence:
            step_result = self.evaluate_step(action_data)
            step_results.append(step_result)
            total_reward += step_result["reward"]["value"]

            if step_result["done"]:
                break

        # Get final state and grade — clamp to strictly (0.001, 0.999)
        final_state = self.environment.state()
        grade = _clamp_score(grade_task(final_state))
        final_score = _clamp_score(final_state.score)

        # Clamp score field inside the serialised state dict as well
        final_state_dict = final_state.dict()
        if "score" in final_state_dict:
            final_state_dict["score"] = _clamp_score(final_state_dict["score"])

        return {
            "setup": setup_info,
            "step_results": step_results,
            "final_state": final_state_dict,
            "grade": grade,
            "final_score": final_score,
            "total_reward": _clamp_score(total_reward),
            "steps_taken": len(step_results),
            "episode_done": final_state.done
        }

    def get_grade(self, environment_state: EnvironmentState) -> float:
        """
        Get grade for a given environment state.

        Args:
            environment_state: Environment state to grade

        Returns:
            Grade strictly between 0.001 and 0.999 (OpenEnv requirement)
        """
        return _clamp_score(grade_task(environment_state))

    def validate_submission(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a submission against requirements.

        Args:
            submission_data: Submission data to validate

        Returns:
            Validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check required fields
        required_fields = ["difficulty", "actions"]
        for field in required_fields:
            if field not in submission_data:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required field: {field}")

        # Validate difficulty
        if "difficulty" in submission_data:
            if submission_data["difficulty"] not in ["easy", "medium", "hard"]:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Invalid difficulty: {submission_data['difficulty']}"
                )

        # Validate actions
        if "actions" in submission_data:
            actions = submission_data["actions"]
            if not isinstance(actions, list):
                validation_results["valid"] = False
                validation_results["errors"].append("Actions must be a list")
            else:
                for i, action in enumerate(actions):
                    if not isinstance(action, dict):
                        validation_results["valid"] = False
                        validation_results["errors"].append(
                            f"Action {i} must be a dictionary"
                        )
                    else:
                        # Validate action structure
                        if "type" not in action:
                            validation_results["valid"] = False
                            validation_results["errors"].append(
                                f"Action {i} missing 'type' field"
                            )
                        else:
                            action_type = action["type"]
                            if action_type not in ["select_station", "wait", "move_to_next_station"]:
                                validation_results["valid"] = False
                                validation_results["errors"].append(
                                    f"Action {i} has invalid type: {action_type}"
                                )

                            # Check required fields for specific action types
                            if action_type == "select_station" and "station_id" not in action:
                                validation_results["valid"] = False
                                validation_results["errors"].append(
                                    f"Action {i} (select_station) missing 'station_id' field"
                                )

        return validation_results


# Global grader instance
_grader_instance = None


def get_grader() -> EVChargingGrader:
    """Get the global grader instance."""
    global _grader_instance
    if _grader_instance is None:
        _grader_instance = EVChargingGrader()
    return _grader_instance


def main():
    """
    Main entry point for command-line grading.

    Usage:
        python grader.py <difficulty> <submission_file>

    Args:
        difficulty: Task difficulty (easy, medium, hard)
        submission_file: JSON file containing submission data
    """
    if len(sys.argv) != 3:
        print("Usage: python grader.py <difficulty> <submission_file>")
        print("Example: python grader.py medium submission.json")
        sys.exit(1)

    difficulty = sys.argv[1]
    submission_file = sys.argv[2]

    try:
        # Load submission
        with open(submission_file, 'r') as f:
            submission_data = json.load(f)

        # Get grader
        grader = get_grader()
        grader.current_difficulty = difficulty

        # Validate submission
        validation_results = grader.validate_submission(submission_data)

        if not validation_results["valid"]:
            print("Submission validation failed:")
            for error in validation_results["errors"]:
                print(f"  ERROR: {error}")
            sys.exit(1)

        if validation_results["warnings"]:
            print("Submission warnings:")
            for warning in validation_results["warnings"]:
                print(f"  WARNING: {warning}")

        # Evaluate episode
        results = grader.evaluate_episode(
            submission_data["actions"],
            seed=submission_data.get("seed", 42)
        )

        # Print results
        print(f"Evaluation Results for {difficulty} task:")
        print(f"  Grade:        {results['grade']:.4f}")
        print(f"  Final Score:  {results['final_score']:.4f}")
        print(f"  Total Reward: {results['total_reward']:.4f}")
        print(f"  Steps Taken:  {results['steps_taken']}")
        print(f"  Episode Done: {results['episode_done']}")

        # Save detailed results
        output_file = submission_file.replace('.json', '_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Detailed results saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Submission file '{submission_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in submission file '{submission_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()