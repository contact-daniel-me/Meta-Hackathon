"""
FastAPI app for Hugging Face Spaces deployment of EV Charging Environment.
Scorched Earth Sanitization (v1.0.1)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import EVChargingEnvironment
from tasks import get_task_config
from models import Action, ActionType


app = FastAPI(
    title="EV Charging Environment",
    description="OpenEnv-compatible EV charging station optimization environment",
    version="1.0.1"
)


# --- Scorched Earth Sanitization Helper ---
def _aggressive_clamp(obj: Any, key: Optional[str] = None) -> Any:
    """
    Ultra-Nuclear Clamping: Forces EVERY numeric value (float or int)
    into the strictly compliant (0.001, 0.999) range.
    No exceptions for coordinates or power.
    """
    if isinstance(obj, dict):
        return {k: _aggressive_clamp(v, k) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_aggressive_clamp(v) for v in obj]
    elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
        val = float(obj)
        if val == 0.0: return 0.001
        if val == 1.0: return 0.999
        return max(0.001, min(0.999, round(val, 4)))
    return obj


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    difficulty: str = "easy"
    action: Dict[str, Any]
    seed: int = 42


# Store environment instances per session
environments = {}


@app.get("/")
async def root():
    return {"message": "EV Charging Environment API", "version": "1.0.1"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset_environment(request: Optional[ResetRequest] = None):
    try:
        if request is None: request = ResetRequest()
        task_config = get_task_config(request.difficulty)
        env = EVChargingEnvironment(task_config, seed=request.seed)
        environments[request.difficulty] = env
        observation = env.reset()
        
        return _aggressive_clamp({
            "observation": observation.dict(),
            "seed": request.seed,
            "info": {}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step_environment(request: StepRequest):
    try:
        if request.difficulty not in environments:
            raise HTTPException(status_code=400, detail="Not initialized")
        
        env = environments[request.difficulty]
        action_data = request.action
        action = Action(
            type=ActionType(action_data["type"]),
            station_id=action_data.get("station_id"),
            wait_time_minutes=action_data.get("wait_time_minutes")
        )
        
        observation, reward, done, info = env.step(action)
        
        return _aggressive_clamp({
            "observation": observation.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/state")
async def get_state(request: Optional[Dict[str, Any]] = None):
    try:
        difficulty = (request or {}).get("difficulty", "easy")
        if difficulty not in environments:
            raise HTTPException(status_code=400, detail="Not initialized")
        env = environments[difficulty]
        state = env.state()
        return _aggressive_clamp(state.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
