"""
Pydantic models for EV Charging Station Environment.

This module defines the data structures used throughout the environment:
- Observation: Current state of the environment
- Action: Agent's decision
- Reward: Feedback signal
- ChargingStation: Station properties
- EV: Electric vehicle properties
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import datetime


class StationStatus(str, Enum):
    """Charging station status."""
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    OUT_OF_SERVICE = "out_of_service"


class ChargingStation(BaseModel):
    """Represents a charging station."""
    id: str = Field(description="Unique station identifier")
    name: str = Field(description="Station name")
    latitude: float = Field(description="Station latitude")
    longitude: float = Field(description="Station longitude")
    power_kw: float = Field(description="Charging power in kW")
    price_per_kwh: float = Field(description="Price per kWh")
    status: StationStatus = Field(description="Current station status")
    waiting_time_minutes: int = Field(description="Estimated waiting time in minutes")
    max_queue_length: int = Field(description="Maximum queue length")
    current_queue_length: int = Field(description="Current queue length")


class EV(BaseModel):
    """Represents an electric vehicle."""
    id: str = Field(description="Unique EV identifier")
    battery_capacity_kwh: float = Field(description="Battery capacity in kWh")
    current_battery_percent: float = Field(description="Current battery level (0-100)")
    consumption_rate_kwh_per_km: float = Field(description="Energy consumption per km")
    priority: int = Field(description="Priority level (1-5, 5=highest)")


class Observation(BaseModel):
    """Environment observation for the agent."""
    ev: EV = Field(description="Current EV state")
    stations: List[ChargingStation] = Field(description="Available charging stations")
    current_location_lat: float = Field(description="Current latitude")
    current_location_lon: float = Field(description="Current longitude")
    destination_lat: float = Field(description="Destination latitude")
    destination_lon: float = Field(description="Destination longitude")
    time_remaining_hours: float = Field(description="Time remaining to reach destination")
    budget_remaining: float = Field(description="Remaining budget for charging")
    step_count: int = Field(description="Current step number")
    max_steps: int = Field(description="Maximum steps allowed")
    other_evs_waiting: List[Dict[str, Any]] = Field(description="Other EVs waiting at stations")


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    SELECT_STATION = "select_station"
    WAIT = "wait"
    MOVE_TO_NEXT_STATION = "move_to_next_station"


class Action(BaseModel):
    """Agent action."""
    type: ActionType = Field(description="Action type")
    station_id: Optional[str] = Field(description="Selected station ID", default=None)
    wait_time_minutes: Optional[int] = Field(description="Wait time in minutes", default=None)


class Reward(BaseModel):
    """Reward signal for the agent."""
    value: float = Field(description="Reward value")
    breakdown: Dict[str, float] = Field(description="Reward component breakdown")
    done: bool = Field(description="Whether episode is done")
    info: Dict[str, Any] = Field(description="Additional information")


class TaskConfig(BaseModel):
    """Configuration for a specific task."""
    name: str = Field(description="Task name")
    difficulty: str = Field(description="Task difficulty: easy, medium, hard")
    description: str = Field(description="Task description")
    max_steps: int = Field(description="Maximum steps per episode")
    num_stations: int = Field(description="Number of charging stations")
    num_other_evs: int = Field(description="Number of other EVs in environment")
    time_limit_hours: float = Field(description="Time limit for reaching destination")
    budget_limit: float = Field(description="Budget limit for charging")
    reward_weights: Dict[str, float] = Field(description="Reward component weights")


class EnvironmentState(BaseModel):
    """Complete environment state."""
    observation: Observation = Field(description="Current observation")
    action_history: List[Action] = Field(description="History of actions taken")
    reward_history: List[float] = Field(description="History of rewards received")
    done: bool = Field(description="Whether episode is finished")
    score: float = Field(description="Current score (0-1)")
    task_config: TaskConfig = Field(description="Current task configuration")
