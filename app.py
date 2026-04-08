"""
FastAPI app for Hugging Face Spaces deployment of EV Charging Environment.
Provides REST API endpoints for OpenEnv compatibility.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import EVChargingEnvironment
from tasks import get_task_config
from models import Action, ActionType


app = FastAPI(
    title="EV Charging Environment",
    description="OpenEnv-compatible EV charging station optimization environment",
    version="1.0.0"
)


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
        "version": "1.0.0",
        "endpoints": {"/reset", "/step", "/state", "/health"}
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset", response_model=ResetResponse)
async def reset_environment(request: Optional[ResetRequest] = None):
    """Reset environment and return initial observation."""
    try:
        if request is None:
            request = ResetRequest()
            
        # Validate difficulty
        if request.difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty")
        
        # Get task configuration
        task_config = get_task_config(request.difficulty)
        
        # Create environment
        env = EVChargingEnvironment(task_config, seed=request.seed)
        
        # Store environment for this session
        environments[request.difficulty] = env
        
        # Reset environment
        observation = env.reset()
        
        # Convert to dict for JSON serialization
        obs_dict = observation.dict()
        
        return ResetResponse(
            observation=obs_dict,
            seed=request.seed,
            info={}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest):
    """Execute one step in the environment."""
    try:
        # Check if environment exists
        if request.difficulty not in environments:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        env = environments[request.difficulty]
        
        # Parse action
        action_data = request.action
        
        # Validate action type
        if "type" not in action_data:
            raise HTTPException(status_code=400, detail="Action must have 'type' field")
        
        action_type = action_data["type"]
        if action_type not in ["select_station", "wait", "move_to_next_station"]:
            raise HTTPException(status_code=400, detail="Invalid action type")
        
        # Create action object
        action = Action(
            type=ActionType(action_type),
            station_id=action_data.get("station_id"),
            wait_time_minutes=action_data.get("wait_time_minutes")
        )
        
        # Execute step
        observation, reward, done, info = env.step(action)
        
        # Convert to dict for JSON serialization
        obs_dict = observation.dict()
        reward_dict = reward.dict()
        
        return StepResponse(
            observation=obs_dict,
            reward=reward_dict,
            done=done,
            info=info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


class StateRequest(BaseModel):
    difficulty: str = "easy"


@app.post("/state")
async def get_state(request: Optional[StateRequest] = None):
    """Get current environment state."""
    try:
        if request is None:
            request = StateRequest()
            
        if request.difficulty not in environments:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        env = environments[request.difficulty]
        state = env.state()
        
        return state.dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}")

@app.get("/state")
async def get_state_get(difficulty: str = "easy"):
    """Get current environment state."""
    try:
        if difficulty not in environments:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        env = environments[difficulty]
        state = env.state()
        
        return state.dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
