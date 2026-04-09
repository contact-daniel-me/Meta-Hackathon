"""
EV Charging Station Environment - OpenEnv Compatible Implementation.

This module implements the core environment with step(), reset(), and state() methods
required by OpenEnv specification.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from models import (
    Observation, Action, Reward, ChargingStation, EV, StationStatus, ActionType,
    TaskConfig, EnvironmentState
)


class EVChargingEnvironment:
    """
    EV Charging Station Selection Environment.
    
    This environment simulates an electric vehicle driver's decision-making process
    when selecting charging stations based on distance, cost, availability, and time constraints.
    """
    
    def __init__(self, task_config: TaskConfig, seed: Optional[int] = None):
        """Initialize the environment."""
        self.task_config = task_config
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Environment state
        self.current_step = 0
        self.ev: Optional[EV] = None
        self.stations: List[ChargingStation] = []
        self.current_location = (0.0, 0.0)
        self.destination = (0.0, 0.0)
        self.time_remaining = task_config.time_limit_hours
        self.budget_remaining = task_config.budget_limit
        self.action_history: List[Action] = []
        self.reward_history: List[float] = []
        self.done = False
        self.score = 0.0
        self.other_evs_waiting: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_distance_traveled = 0.0
        self.total_waiting_time = 0.0
        self.total_charging_cost = 0.0
        self.battery_charged = 0.0
        
    def reset(self) -> Observation:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.action_history = []
        self.reward_history = []
        self.done = False
        self.score = 0.0
        self.total_distance_traveled = 0.0
        self.total_waiting_time = 0.0
        self.total_charging_cost = 0.0
        self.battery_charged = 0.0
        
        # Generate EV
        self.ev = self._generate_ev()
        
        # Generate charging stations
        self.stations = self._generate_stations()
        
        # Generate start and destination locations
        self.current_location = self._generate_location()
        self.destination = self._generate_location()
        
        # Reset time and budget
        self.time_remaining = self.task_config.time_limit_hours
        self.budget_remaining = self.task_config.budget_limit
        
        # Generate other EVs
        self.other_evs_waiting = self._generate_other_evs()
        
        return self._get_observation()
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        
        self.action_history.append(action)
        self.current_step += 1
        
        # Process action
        reward_value, reward_breakdown, info = self._process_action(action)
        
        # Update environment state
        self._update_environment_state(action, info)
        
        # Check if episode is done
        self.done = self._check_done()
        
        # Calculate final score if done
        if self.done:
            self.score = self._calculate_final_score()
        
        # Create reward object
        reward = Reward(
            value=reward_value,
            breakdown=reward_breakdown,
            done=self.done,
            info=info
        )
        
        self.reward_history.append(reward_value)
        
        observation = self._get_observation()
        
        return observation, reward, self.done, info
    
    def state(self) -> EnvironmentState:
        """
        Get current environment state.
        
        Returns:
            Current environment state
        """
        return EnvironmentState(
            observation=self._get_observation(),
            action_history=self.action_history.copy(),
            reward_history=self.reward_history.copy(),
            done=self.done,
            score=self.score,
            task_config=self.task_config
        )
    
    def _generate_ev(self) -> EV:
        """Generate a random EV."""
        battery_capacities = [40, 60, 75, 100, 120]  # kWh
        consumption_rates = [0.15, 0.18, 0.20, 0.22, 0.25]  # kWh/km
        
        return EV(
            id=f"ev_{self.rng.randint(1000, 9999)}",
            battery_capacity_kwh=self.rng.choice(battery_capacities),
            current_battery_percent=self.rng.uniform(20, 80),
            consumption_rate_kwh_per_km=self.rng.choice(consumption_rates),
            priority=self.rng.randint(1, 5)
        )
    
    def _generate_stations(self) -> List[ChargingStation]:
        """Generate charging stations."""
        stations = []
        
        for i in range(self.task_config.num_stations):
            # Random location around current area
            lat = self.rng.uniform(-0.1, 0.1)
            lon = self.rng.uniform(-0.1, 0.1)
            
            # Station properties
            power_options = [22, 50, 150, 350]  # kW
            price_range = (0.15, 0.45)  # $/kWh
            
            # Random status with bias towards available
            status_weights = [0.6, 0.3, 0.1]  # available, occupied, out_of_service
            status = self.rng.choices(
                [StationStatus.AVAILABLE, StationStatus.OCCUPIED, StationStatus.OUT_OF_SERVICE],
                weights=status_weights
            )[0]
            
            # Waiting time based on status
            if status == StationStatus.AVAILABLE:
                waiting_time = 0
                queue_length = 0
            elif status == StationStatus.OCCUPIED:
                waiting_time = self.rng.randint(10, 60)
                queue_length = self.rng.randint(1, 5)
            else:
                waiting_time = 999  # Very high for out of service
                queue_length = 0
            
            station = ChargingStation(
                id=f"station_{i+1}",
                name=f"Charging Station {i+1}",
                latitude=lat,
                longitude=lon,
                power_kw=self.rng.choice(power_options),
                price_per_kwh=self.rng.uniform(*price_range),
                status=status,
                waiting_time_minutes=waiting_time,
                max_queue_length=self.rng.randint(3, 8),
                current_queue_length=queue_length
            )
            stations.append(station)
        
        return stations
    
    def _generate_location(self) -> Tuple[float, float]:
        """Generate a random location."""
        return (self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.05, 0.05))
    
    def _generate_other_evs(self) -> List[Dict[str, Any]]:
        """Generate other EVs waiting at stations."""
        other_evs = []
        
        for _ in range(self.task_config.num_other_evs):
            station_idx = self.rng.randint(0, len(self.stations) - 1)
            station = self.stations[station_idx]
            
            if station.status == StationStatus.AVAILABLE:
                continue
            
            other_ev = {
                "id": f"other_ev_{self.rng.randint(1000, 9999)}",
                "station_id": station.id,
                "arrival_time": self.rng.randint(0, 120),  # minutes ago
                "priority": self.rng.randint(1, 5),
                "battery_percent": self.rng.uniform(5, 30)
            }
            other_evs.append(other_ev)
        
        return other_evs
    
    def _get_observation(self) -> Observation:
        """Get current observation."""
        return Observation(
            ev=self.ev,
            stations=self.stations,
            current_location_lat=self.current_location[0],
            current_location_lon=self.current_location[1],
            destination_lat=self.destination[0],
            destination_lon=self.destination[1],
            time_remaining_hours=self.time_remaining,
            budget_remaining=self.budget_remaining,
            step_count=self.current_step,
            max_steps=self.task_config.max_steps,
            other_evs_waiting=self.other_evs_waiting
        )
    
    def _process_action(self, action: Action) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Process action and return reward."""
        reward_breakdown = {}
        info = {}
        
        if action.type == ActionType.SELECT_STATION:
            reward_value, reward_breakdown, info = self._handle_select_station(action)
        elif action.type == ActionType.WAIT:
            reward_value, reward_breakdown, info = self._handle_wait(action)
        elif action.type == ActionType.MOVE_TO_NEXT_STATION:
            reward_value, reward_breakdown, info = self._handle_move_to_next()
        else:
            # Invalid action
            reward_value = -0.1
            reward_breakdown["invalid_action"] = -0.1
            info["error"] = "Invalid action type"
        
        return reward_value, reward_breakdown, info
    
    def _handle_select_station(self, action: Action) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Handle station selection action."""
        info = {}
        station_id = action.station_id
        if not station_id:
            return -0.05, {"no_station_selected": -0.05}, {"error": "No station selected"}
        
        # Find station
        station = None
        for s in self.stations:
            if s.id == station_id:
                station = s
                break
        
        if not station:
            return -0.05, {"station_not_found": -0.05}, {"error": "Station not found"}
        
        # Calculate distance
        distance_km = self._calculate_distance(
            self.current_location, (station.latitude, station.longitude)
        )
        
        # Calculate rewards and penalties
        reward_breakdown = {}
        reward_value = 0.0
        
        # Distance reward (closer is better)
        distance_reward = -0.01 * min(distance_km, 50)  # Cap at 50km
        reward_breakdown["distance_penalty"] = distance_reward
        reward_value += distance_reward
        
        # Station availability reward
        if station.status == StationStatus.AVAILABLE:
            availability_reward = 0.1
            reward_breakdown["station_available"] = availability_reward
            reward_value += availability_reward
            
            # Simulate charging
            self._simulate_charging(station)
            info["charging_completed"] = True
            
        elif station.status == StationStatus.OCCUPIED:
            # Waiting penalty
            waiting_penalty = -0.001 * station.waiting_time_minutes
            reward_breakdown["waiting_penalty"] = waiting_penalty
            reward_value += waiting_penalty
            
            self.total_waiting_time += station.waiting_time_minutes
            self.time_remaining -= station.waiting_time_minutes / 60
            info["waiting_time"] = station.waiting_time_minutes
            
        else:  # OUT_OF_SERVICE
            penalty = -0.2
            reward_breakdown["station_out_of_service"] = penalty
            reward_value += penalty
            info["station_unavailable"] = True
        
        # Cost consideration
        cost_penalty = -0.0001 * station.price_per_kwh
        reward_breakdown["cost_penalty"] = cost_penalty
        reward_value += cost_penalty
        
        # Update location
        self.current_location = (station.latitude, station.longitude)
        self.total_distance_traveled += distance_km
        
        info.update({
            "station_id": station_id,
            "distance_km": distance_km,
            "station_status": station.status,
            "price_per_kwh": station.price_per_kwh
        })
        
        return reward_value, reward_breakdown, info
    
    def _handle_wait(self, action: Action) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Handle wait action."""
        wait_time = action.wait_time_minutes or 10
        
        # Waiting penalty
        penalty = -0.0001 * wait_time
        reward_breakdown = {"waiting_penalty": penalty}
        
        self.total_waiting_time += wait_time
        self.time_remaining -= wait_time / 60
        
        # Update station statuses (some might become available)
        self._update_station_statuses()
        
        info = {
            "wait_time_minutes": wait_time,
            "time_remaining": self.time_remaining
        }
        
        return penalty, reward_breakdown, info
    
    def _handle_move_to_next(self) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Handle move to next station action."""
        # Find nearest available station
        available_stations = [
            s for s in self.stations if s.status == StationStatus.AVAILABLE
        ]
        
        info = {}
        
        if not available_stations:
            return -0.05, {"no_available_stations": -0.05}, {"error": "No available stations"}
        
        # Find nearest
        nearest_station = min(
            available_stations,
            key=lambda s: self._calculate_distance(
                self.current_location, (s.latitude, s.longitude)
            )
        )
        
        distance_km = self._calculate_distance(
            self.current_location, (nearest_station.latitude, nearest_station.longitude)
        )
        
        # Small penalty for indecision
        penalty = -0.01
        reward_breakdown = {"indecision_penalty": penalty}
        
        self.current_location = (nearest_station.latitude, nearest_station.longitude)
        self.total_distance_traveled += distance_km
        
        info = {
            "moved_to_station": nearest_station.id,
            "distance_km": distance_km
        }
        
        return penalty, reward_breakdown, info
    
    def _simulate_charging(self, station: ChargingStation):
        """Simulate charging at a station."""
        if not self.ev:
            return
        
        # Calculate how much energy needed
        energy_needed = self.ev.battery_capacity_kwh * (1 - self.ev.current_battery_percent / 100)
        
        # Calculate charging time (simplified)
        charging_time_hours = energy_needed / station.power_kw
        
        # Update EV battery
        self.ev.current_battery_percent = min(100, self.ev.current_battery_percent + 50)
        
        # Update budget and time
        charging_cost = energy_needed * station.price_per_kwh
        self.total_charging_cost += charging_cost
        self.budget_remaining -= charging_cost
        self.time_remaining -= charging_time_hours
        
        # Update station status to occupied
        station.status = StationStatus.OCCUPIED
        station.waiting_time_minutes = self.rng.randint(10, 30)
    
    def _update_station_statuses(self):
        """Randomly update station statuses to simulate dynamic environment."""
        for station in self.stations:
            # Small chance of status change
            if self.rng.random() < 0.1:
                if station.status == StationStatus.OCCUPIED:
                    if self.rng.random() < 0.3:
                        station.status = StationStatus.AVAILABLE
                        station.waiting_time_minutes = 0
                        station.current_queue_length = 0
                elif station.status == StationStatus.OUT_OF_SERVICE:
                    if self.rng.random() < 0.1:
                        station.status = StationStatus.AVAILABLE
                        station.waiting_time_minutes = 0
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations in km."""
        # Simplified distance calculation (for small distances)
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        # Convert to meters (approximate)
        lat_diff = (lat2 - lat1) * 111320  # meters per degree latitude
        lon_diff = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))  # meters per degree longitude
        
        distance_meters = math.sqrt(lat_diff**2 + lon_diff**2)
        return distance_meters / 1000  # Convert to km
    
    def _update_environment_state(self, action: Action, info: Dict[str, Any]):
        """Update environment state after action."""
        # Update other EVs
        self.other_evs_waiting = [ev for ev in self.other_evs_waiting if ev["arrival_time"] < 180]
        
        # Add some randomness to other EVs
        if self.rng.random() < 0.2:
            new_ev = self._generate_other_evs()
            if new_ev:
                self.other_evs_waiting.extend(new_ev[:1])
    
    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Check step limit
        if self.current_step >= self.task_config.max_steps:
            return True
        
        # Check time limit
        if self.time_remaining <= 0:
            return True
        
        # Check budget limit
        if self.budget_remaining <= 0:
            return True
        
        # Check if EV is fully charged and can reach destination
        if self.ev and self.ev.current_battery_percent >= 80:
            distance_to_destination = self._calculate_distance(
                self.current_location, self.destination
            )
            max_range = (self.ev.current_battery_percent / 100) * self.ev.battery_capacity_kwh / self.ev.consumption_rate_kwh_per_km
            
            if max_range >= distance_to_destination * 1.2:  # 20% buffer
                return True
        
        return False
    
    def _calculate_final_score(self) -> float:
        """Calculate final score (0-1)."""
        score_components = {}
        
        # Battery level score (0-0.3)
        if self.ev:
            battery_score = min(0.3, self.ev.current_battery_percent / 100 * 0.3)
            score_components["battery"] = battery_score
        
        # Time efficiency score (0-0.2)
        time_used = self.task_config.time_limit_hours - self.time_remaining
        time_efficiency = max(0, 1 - (time_used / self.task_config.time_limit_hours))
        score_components["time"] = time_efficiency * 0.2
        
        # Cost efficiency score (0-0.2)
        budget_used = self.task_config.budget_limit - self.budget_remaining
        if self.task_config.budget_limit > 0:
            cost_efficiency = max(0, 1 - (budget_used / self.task_config.budget_limit))
            score_components["cost"] = cost_efficiency * 0.2
        else:
            score_components["cost"] = 0.2
        
        # Distance efficiency score (0-0.2)
        if self.total_distance_traveled > 0:
            # Penalize excessive travel
            distance_penalty = min(0.2, self.total_distance_traveled / 100 * 0.02)
            score_components["distance"] = max(0, 0.2 - distance_penalty)
        else:
            score_components["distance"] = 0.1
        
        # Decision quality score (0-0.1)
        if self.action_history:
            # Reward for good decisions
            good_decisions = sum(1 for r in self.reward_history if r > 0)
            decision_score = min(0.1, good_decisions / len(self.action_history) * 0.1)
            score_components["decisions"] = decision_score
        else:
            score_components["decisions"] = 0
        
        total_score = sum(score_components.values())
        # Ensure score is strictly between 0 and 1 (OpenEnv requirement)
        return min(0.999, max(0.001, total_score))
