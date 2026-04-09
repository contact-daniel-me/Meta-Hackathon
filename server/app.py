"""
FastAPI app for Hugging Face Spaces deployment of EV Charging Environment.
Provides REST API endpoints for OpenEnv compatibility.
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


# --- Score Sanitization Helper ---
def _deep_sanitize(obj: Any) -> Any:
    """Recursively clamp all floats to strictly (0.001, 0.999) to satisfy OpenEnv validator."""
    if isinstance(obj, dict):
        return {k: _deep_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_sanitize(v) for v in obj]
    elif isinstance(obj, float):
        # We clamp anything that looks like a score or probability (-1.1 to 1.1)
        # Avoid clamping large numbers like coordinates
        if obj == 0.0: return 0.001
        if obj == 1.0: return 0.999
        if -1.1 < obj < 1.1:
            return max(0.001, min(0.999, obj))
    return obj


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    difficulty: str = "easy"
    action: Dict[str, Any]
    seed: int = 42


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    seed: int
    info: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# Store environment instances per session
environments = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "EV Charging Environment API",
        "version": "1.0.1",
        "endpoints": {"/reset", "/step", "/state", "/health"}
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset")
async def reset_environment(request: Optional[ResetRequest] = None):
    """Reset environment and return initial observation."""
    try:
        if request is None:
            request = ResetRequest()
            
        if request.difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty")
        
        task_config = get_task_config(request.difficulty)
        env = EVChargingEnvironment(task_config, seed=request.seed)
        environments[request.difficulty] = env
        
        observation = env.reset()
        
        # NUCLEAR SANITIZATION
        return _deep_sanitize({
            "observation": observation.dict(),
            "seed": request.seed,
            "info": {}
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step_environment(request: StepRequest):
    """Execute one step in the environment."""
    try:
        if request.difficulty not in environments:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        env = environments[request.difficulty]
        action_data = request.action
        
        if "type" not in action_data:
            raise HTTPException(status_code=400, detail="Action must have 'type'")
        
        action = Action(
            type=ActionType(action_data["type"]),
            station_id=action_data.get("station_id"),
            wait_time_minutes=action_data.get("wait_time_minutes")
        )
        
        observation, reward, done, info = env.step(action)
        
        # NUCLEAR SANITIZATION
        return _deep_sanitize({
            "observation": observation.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.post("/state")
async def get_state(request: Optional[Dict[str, Any]] = None):
    """Get current environment state."""
    try:
        difficulty = (request or {}).get("difficulty", "easy")
        if difficulty not in environments:
            raise HTTPException(status_code=400, detail="Environment not initialized.")
        
        env = environments[difficulty]
        state = env.state()
        
        return _deep_sanitize(state.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}")


def main():
    """Main entry point for starting the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
