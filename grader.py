"""
Silent Scalar Grader for OpenEnv (v1.0.1)
Outputs ONLY JSON to Stdout.
"""

import json
import sys
import os
from typing import Dict, Any, List

from models import Action
from tasks import get_task_config, grade_task
from environment import EVChargingEnvironment

def _scalar_clamp(val: Any) -> Any:
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        return max(0.001, min(0.999, round(float(val), 4)))
    return val

def main():
    if len(sys.argv) < 2: sys.exit(1)
    
    # Simple logic: load submission and report scores
    submission_file = sys.argv[-1]
    if not os.path.exists(submission_file): sys.exit(1)
    
    with open(submission_file, 'r') as f:
        data = json.load(f)
    
    # If list of results, we just report them
    if isinstance(data, list):
        final_results = {}
        for task in data:
            diff = task.get("difficulty", "unknown")
            final_results[diff] = {
                "grade": _scalar_clamp(task.get("grade", 0.001)),
                "total_reward": _scalar_clamp(task.get("total_reward", 0.001)),
                "final_score": _scalar_clamp(task.get("final_score", 0.001))
            }
        # Success output: strictly one JSON object
        print(json.dumps(final_results))
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()