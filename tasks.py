"""
Task definitions and deterministic graders for EV Charging Environment.

This module defines three difficulty levels with corresponding tasks and grader functions.
Each task has a clear objective and returns a score between 0.0 and 1.0.
"""

from typing import Dict, List, Any, Tuple
import math
from dataclasses import dataclass

from models import TaskConfig, EnvironmentState, Observation, Action
from environment import EVChargingEnvironment


class EVChargingTasks:
    """Collection of EV charging tasks with different difficulty levels."""
    
    @staticmethod
    def get_easy_task() -> TaskConfig:
        """
        Easy Task: Choose the nearest charging station.
        
        Objective: Select the closest available charging station to minimize travel distance.
        The agent should prioritize proximity over other factors.
        """
        return TaskConfig(
            name="Nearest Station Selection",
            difficulty="easy",
            description="Select the nearest available charging station to minimize travel distance.",
            max_steps=10,
            num_stations=5,
            num_other_evs=2,
            time_limit_hours=4.0,
            budget_limit=50.0,
            reward_weights={
                "distance": 0.6,
                "availability": 0.3,
                "completion": 0.1
            }
        )
    
    @staticmethod
    def get_medium_task() -> TaskConfig:
        """
        Medium Task: Optimize charging based on cost and waiting time.
        
        Objective: Balance between distance, cost, and waiting time to find the optimal station.
        The agent should consider multiple factors and make trade-offs.
        """
        return TaskConfig(
            name="Cost and Time Optimization",
            difficulty="medium",
            description="Optimize charging station selection considering cost, waiting time, and distance.",
            max_steps=15,
            num_stations=8,
            num_other_evs=5,
            time_limit_hours=3.0,
            budget_limit=40.0,
            reward_weights={
                "cost_efficiency": 0.3,
                "time_efficiency": 0.3,
                "distance": 0.2,
                "availability": 0.2
            }
        )
    
    @staticmethod
    def get_hard_task() -> TaskConfig:
        """
        Hard Task: Multi-user scheduling with limited stations and priorities.
        
        Objective: Navigate complex scenarios with multiple competing EVs, limited resources,
        and priority-based scheduling. The agent must make strategic decisions under pressure.
        """
        return TaskConfig(
            name="Multi-user Priority Scheduling",
            difficulty="hard",
            description="Manage charging in a competitive environment with limited stations and priority-based access.",
            max_steps=20,
            num_stations=6,
            num_other_evs=10,
            time_limit_hours=2.0,
            budget_limit=30.0,
            reward_weights={
                "priority_handling": 0.3,
                "resource_efficiency": 0.25,
                "time_management": 0.25,
                "cost_optimization": 0.2
            }
        )


class EVChargingGraders:
    """Deterministic graders for evaluating task performance."""
    
    @staticmethod
    def easy_grader(environment_state: EnvironmentState) -> float:
        """
        Grader for Easy Task: Nearest Station Selection.
        
        Evaluation criteria:
        - Selected station proximity (40%)
        - Success in finding available station (30%)
        - Episode completion efficiency (20%)
        - Minimal unnecessary actions (10%)
        """
        if not environment_state.action_history:
            return 0.0
        
        score = 0.0
        
        # Extract relevant data from final observation
        obs = environment_state.observation
        stations = obs.stations
        current_location = (obs.current_location_lat, obs.current_location_lon)
        
        # 1. Proximity score (40%)
        proximity_score = 0.0
        if environment_state.action_history:
            last_action = environment_state.action_history[-1]
            if last_action.station_id:
                # Find the selected station
                selected_station = None
                for station in stations:
                    if station.id == last_action.station_id:
                        selected_station = station
                        break
                
                if selected_station:
                    # Calculate distance to selected station
                    distance = EVChargingGraders._calculate_distance(
                        current_location, (selected_station.latitude, selected_station.longitude)
                    )
                    
                    # Find nearest available station for comparison
                    available_stations = [s for s in stations if s.status.value == "available"]
                    if available_stations:
                        nearest_distance = min(
                            EVChargingGraders._calculate_distance(
                                current_location, (s.latitude, s.longitude)
                            ) for s in available_stations
                        )
                        
                        # Score based on how close to optimal
                        if distance <= nearest_distance * 1.1:  # Within 10% of optimal
                            proximity_score = 0.4
                        elif distance <= nearest_distance * 1.5:  # Within 50% of optimal
                            proximity_score = 0.3
                        elif distance <= nearest_distance * 2.0:  # Within 100% of optimal
                            proximity_score = 0.2
                        else:
                            proximity_score = 0.1
        
        score += proximity_score
        
        # 2. Availability success (30%)
        availability_score = 0.0
        if environment_state.action_history:
            last_action = environment_state.action_history[-1]
            if last_action.station_id:
                for station in stations:
                    if station.id == last_action.station_id:
                        if station.status.value == "available":
                            availability_score = 0.3
                        elif station.status.value == "occupied":
                            availability_score = 0.15
                        break
        
        score += availability_score
        
        # 3. Completion efficiency (20%)
        completion_score = 0.0
        if environment_state.done:
            if obs.ev.current_battery_percent >= 80:
                completion_score = 0.2
            elif obs.ev.current_battery_percent >= 50:
                completion_score = 0.1
        else:
            # Partial credit for progress
            if obs.ev.current_battery_percent > obs.ev.current_battery_percent:
                completion_score = 0.05
        
        score += completion_score
        
        # 4. Action efficiency (10%)
        action_efficiency = 0.0
        if len(environment_state.action_history) <= 5:
            action_efficiency = 0.1
        elif len(environment_state.action_history) <= 8:
            action_efficiency = 0.05
        
        score += action_efficiency
        
        # Ensure score is strictly between 0 and 1 (OpenEnv requirement)
        return min(0.999, max(0.001, score))
    
    @staticmethod
    def medium_grader(environment_state: EnvironmentState) -> float:
        """
        Grader for Medium Task: Cost and Time Optimization.
        
        Evaluation criteria:
        - Cost efficiency (30%)
        - Time efficiency (30%)
        - Distance consideration (20%)
        - Waiting time management (20%)
        """
        if not environment_state.action_history:
            return 0.0
        
        score = 0.0
        obs = environment_state.observation
        task_config = environment_state.task_config
        
        # 1. Cost efficiency (30%)
        cost_score = 0.0
        budget_used = task_config.budget_limit - obs.budget_remaining
        if budget_used > 0:
            cost_efficiency = 1 - (budget_used / task_config.budget_limit)
            cost_score = cost_efficiency * 0.3
        else:
            cost_score = 0.3  # Perfect if no budget used
        
        score += cost_score
        
        # 2. Time efficiency (30%)
        time_score = 0.0
        time_used = task_config.time_limit_hours - obs.time_remaining_hours
        if time_used > 0:
            time_efficiency = 1 - (time_used / task_config.time_limit_hours)
            time_score = time_efficiency * 0.3
        else:
            time_score = 0.3  # Perfect if no time used
        
        score += time_score
        
        # 3. Distance consideration (20%)
        distance_score = 0.0
        if environment_state.action_history:
            # Analyze if agent considered distance appropriately
            total_distance = 0.0
            for action in environment_state.action_history:
                if action.station_id:
                    for station in obs.stations:
                        if station.id == action.station_id:
                            # This is simplified - in practice, we'd track position changes
                            distance_score += 0.02
                            break
            
            distance_score = min(0.2, distance_score)
        
        score += distance_score
        
        # 4. Waiting time management (20%)
        waiting_score = 0.0
        total_waiting = 0
        for action in environment_state.action_history:
            if action.type.value == "wait":
                total_waiting += action.wait_time_minutes or 10
        
        # Lower waiting time is better
        if total_waiting == 0:
            waiting_score = 0.2
        elif total_waiting <= 30:
            waiting_score = 0.15
        elif total_waiting <= 60:
            waiting_score = 0.1
        else:
            waiting_score = 0.05
        
        score += waiting_score
        
        # Ensure score is strictly between 0 and 1 (OpenEnv requirement)
        return min(0.999, max(0.001, score))
    
    @staticmethod
    def hard_grader(environment_state: EnvironmentState) -> float:
        """
        Grader for Hard Task: Multi-user Priority Scheduling.
        
        Evaluation criteria:
        - Priority-based decision making (30%)
        - Resource efficiency under competition (25%)
        - Time management under pressure (25%)
        - Strategic adaptation (20%)
        """
        if not environment_state.action_history:
            return 0.0
        
        score = 0.0
        obs = environment_state.observation
        task_config = environment_state.task_config
        
        # 1. Priority handling (30%)
        priority_score = 0.0
        ev_priority = obs.ev.priority
        
        # Check if agent made priority-appropriate decisions
        high_priority_decisions = 0
        total_decisions = 0
        
        for action in environment_state.action_history:
            if action.station_id:
                total_decisions += 1
                for station in obs.stations:
                    if station.id == action.station_id:
                        # High priority EV should be more aggressive
                        if ev_priority >= 4 and station.status.value == "available":
                            high_priority_decisions += 1
                        # Low priority EV should be more patient
                        elif ev_priority <= 2 and action.type.value == "wait":
                            high_priority_decisions += 1
                        break
        
        if total_decisions > 0:
            priority_ratio = high_priority_decisions / total_decisions
            priority_score = priority_ratio * 0.3
        
        score += priority_score
        
        # 2. Resource efficiency (25%)
        resource_score = 0.0
        other_evs_count = len(obs.other_evs_waiting)
        
        # Success in competitive environment
        if environment_state.done and obs.ev.current_battery_percent >= 50:
            if other_evs_count >= 8:
                resource_score = 0.25  # High competition
            elif other_evs_count >= 5:
                resource_score = 0.2  # Medium competition
            else:
                resource_score = 0.15  # Low competition
        
        score += resource_score
        
        # 3. Time management under pressure (25%)
        time_pressure_score = 0.0
        time_remaining_ratio = obs.time_remaining_hours / task_config.time_limit_hours
        
        if time_remaining_ratio >= 0.5:
            time_pressure_score = 0.25
        elif time_remaining_ratio >= 0.25:
            time_pressure_score = 0.15
        elif time_remaining_ratio >= 0.1:
            time_pressure_score = 0.1
        else:
            time_pressure_score = 0.05
        
        score += time_pressure_score
        
        # 4. Strategic adaptation (20%)
        adaptation_score = 0.0
        
        # Check if agent adapted to changing conditions
        action_variety = set()
        for action in environment_state.action_history:
            action_variety.add(action.type.value)
        
        if len(action_variety) >= 3:
            adaptation_score = 0.2  # Used multiple strategies
        elif len(action_variety) >= 2:
            adaptation_score = 0.15  # Some adaptation
        else:
            adaptation_score = 0.05  # Minimal adaptation
        
        score += adaptation_score
        
        # Ensure score is strictly between 0 and 1 (OpenEnv requirement)
        return min(0.999, max(0.001, score))
    
    @staticmethod
    def _calculate_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations in km."""
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        # Simplified distance calculation
        lat_diff = (lat2 - lat1) * 111320  # meters per degree latitude
        lon_diff = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
        
        distance_meters = math.sqrt(lat_diff**2 + lon_diff**2)
        return distance_meters / 1000  # Convert to km


def get_task_config(difficulty: str) -> TaskConfig:
    """Get task configuration by difficulty level."""
    tasks = EVChargingTasks()
    
    if difficulty == "easy":
        return tasks.get_easy_task()
    elif difficulty == "medium":
        return tasks.get_medium_task()
    elif difficulty == "hard":
        return tasks.get_hard_task()
    else:
        raise ValueError(f"Unknown difficulty level: {difficulty}")


def grade_task(environment_state: EnvironmentState) -> float:
    """Grade task performance based on difficulty level."""
    graders = EVChargingGraders()
    difficulty = environment_state.task_config.difficulty
    
    if difficulty == "easy":
        return graders.easy_grader(environment_state)
    elif difficulty == "medium":
        return graders.medium_grader(environment_state)
    elif difficulty == "hard":
        return graders.hard_grader(environment_state)
    else:
        raise ValueError(f"Unknown difficulty level: {difficulty}")


def run_task_episode(difficulty: str, seed: int = 42) -> Dict[str, Any]:
    """
    Run a single episode of the specified task.
    
    Args:
        difficulty: Task difficulty level
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with episode results
    """
    # Get task configuration
    task_config = get_task_config(difficulty)
    
    # Create environment
    env = EVChargingEnvironment(task_config, seed=seed)
    
    # Reset environment
    observation = env.reset()
    
    # Run episode (simplified - in practice, this would use an agent)
    done = False
    steps = 0
    
    while not done and steps < task_config.max_steps:
        # Simple random action for demonstration
        import random
        action_type = random.choice(["select_station", "wait", "move_to_next_station"])
        
        if action_type == "select_station" and observation.stations:
            station = random.choice(observation.stations)
            action = Action(
                type="select_station",
                station_id=station.id
            )
        elif action_type == "wait":
            action = Action(
                type="wait",
                wait_time_minutes=random.randint(5, 30)
            )
        else:
            action = Action(type="move_to_next_station")
        
        observation, reward, done, info = env.step(action)
        steps += 1
    
    # Get final state and grade
    final_state = env.state()
    grade = grade_task(final_state)
    
    return {
        "difficulty": difficulty,
        "steps": steps,
        "final_score": final_state.score,
        "grade": grade,
        "done": done,
        "final_battery": final_state.observation.ev.current_battery_percent,
        "time_remaining": final_state.observation.time_remaining_hours,
        "budget_remaining": final_state.observation.budget_remaining
    }
