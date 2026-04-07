"""
FastAPI app for Hugging Face Spaces deployment of EV Charging Environment.
Provides REST API endpoints for OpenEnv compatibility.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import EVChargingEnvironment
from tasks import get_task_config, grade_task
from models import Action, ActionType, Observation


app = FastAPI(
    title="EV Charging Environment",
    description="OpenEnv-compatible EV charging station optimization environment",
    version="1.0.0"
)


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""
    difficulty: str
    seed: int = 42


class StepRequest(BaseModel):
    """Request model for step endpoint."""
    difficulty: str
    action: Dict[str, Any]
    seed: int = 42


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    observation: Dict[str, Any]
    task_config: Dict[str, Any]
    seed: int


class StepResponse(BaseModel):
    """Response model for step endpoint."""
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# Global environment instances (one per difficulty)
environments: Dict[str, EVChargingEnvironment] = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "EV Charging Environment API",
        "version": "1.0.0",
        "endpoints": {
            "reset": "/reset",
            "step": "/step",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ev-charging-environment"}


@app.post("/reset", response_model=ResetResponse)
async def reset_environment(request: ResetRequest):
    """Reset environment and return initial observation."""
    try:
        # Validate difficulty
        if request.difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty. Must be: easy, medium, hard")
        
        # Get task configuration
        task_config = get_task_config(request.difficulty)
        
        # Create environment
        env = EVChargingEnvironment(task_config, seed=request.seed)
        environments[request.difficulty] = env
        
        # Reset environment
        observation = env.reset()
        
        return ResetResponse(
            observation=observation.dict(),
            task_config=task_config.dict(),
            seed=request.seed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest):
    """Execute one step in the environment."""
    try:
        # Validate difficulty
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
        
        return StepResponse(
            observation=observation.dict(),
            reward=reward.dict(),
            done=done,
            info=info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/tasks")
async def get_tasks():
    """Get available tasks and their configurations."""
    tasks_info = {}
    for difficulty in ["easy", "medium", "hard"]:
        task_config = get_task_config(difficulty)
        tasks_info[difficulty] = task_config.dict()
    
    return {"tasks": tasks_info}


@app.get("/grade")
async def grade_submission(difficulty: str, episode_data: Dict[str, Any] = None):
    """Grade a completed episode (simplified for demo)."""
    try:
        if difficulty not in environments:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        env = environments[difficulty]
        state = env.state()
        grade = grade_task(state)
        
        return {
            "difficulty": difficulty,
            "grade": grade,
            "score": state.score,
            "done": state.done
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run on port 7860 for Hugging Face Spaces
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )
