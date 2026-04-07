#!/usr/bin/env python3
"""
Demonstration script for EV Charging Environment.
Shows the environment in action without requiring an API key.
"""

from environment import EVChargingEnvironment
from tasks import get_task_config
from models import Action, ActionType
import json


def demo_easy_task():
    """Demonstrate the easy task with a simple agent."""
    print("🚗 EV Charging Environment Demo")
    print("=" * 50)
    
    # Create environment
    task_config = get_task_config('easy')
    env = EVChargingEnvironment(task_config, seed=42)
    
    # Reset environment
    observation = env.reset()
    
    print(f"\n📍 Initial State:")
    print(f"   EV Battery: {observation.ev.current_battery_percent:.1f}%")
    print(f"   EV Priority: {observation.ev.priority}")
    print(f"   Time Remaining: {observation.time_remaining_hours:.1f} hours")
    print(f"   Budget Remaining: ${observation.budget_remaining:.2f}")
    print(f"   Available Stations: {len(observation.stations)}")
    
    # Show available stations
    print(f"\n⚡ Available Charging Stations:")
    for i, station in enumerate(observation.stations[:3]):  # Show first 3
        print(f"   {station.id}: {station.name}")
        print(f"      Status: {station.status}")
        print(f"      Power: {station.power_kw} kW")
        print(f"      Price: ${station.price_per_kwh:.2f}/kWh")
        print(f"      Waiting: {station.waiting_time_minutes} min")
    
    # Simple agent: choose nearest available station
    print(f"\n🤖 Agent Decision:")
    available_stations = [s for s in observation.stations if s.status.value == "available"]
    
    if available_stations:
        # Find nearest (simplified - just pick first available)
        chosen_station = available_stations[0]
        action = Action(
            type=ActionType.SELECT_STATION,
            station_id=chosen_station.id
        )
        
        print(f"   Choosing: {chosen_station.id}")
        print(f"   Reason: First available station")
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        print(f"\n📊 Action Result:")
        print(f"   Reward: {reward.value:.4f}")
        print(f"   Done: {done}")
        print(f"   Info: {info}")
        
        if 'charging_completed' in info:
            print(f"   ✅ Charging completed!")
        
        print(f"\n🔋 Final State:")
        print(f"   EV Battery: {obs.ev.current_battery_percent:.1f}%")
        print(f"   Time Remaining: {obs.time_remaining_hours:.1f} hours")
        print(f"   Budget Remaining: ${obs.budget_remaining:.2f}")
        
        # Get final score
        final_state = env.state()
        print(f"   Final Score: {final_state.score:.3f}")
        
    else:
        print("   ❌ No available stations found!")
        action = Action(type=ActionType.WAIT, wait_time_minutes=10)
        print(f"   Waiting 10 minutes for stations to become available...")


def demo_all_difficulties():
    """Demonstrate all three difficulty levels."""
    print("\n\n🎯 Testing All Difficulty Levels")
    print("=" * 50)
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n📈 {difficulty.upper()} Task:")
        
        task_config = get_task_config(difficulty)
        env = EVChargingEnvironment(task_config, seed=42)
        observation = env.reset()
        
        print(f"   Stations: {task_config.num_stations}")
        print(f"   Other EVs: {task_config.num_other_evs}")
        print(f"   Time Limit: {task_config.time_limit_hours} hours")
        print(f"   Budget: ${task_config.budget_limit}")
        print(f"   Max Steps: {task_config.max_steps}")
        
        # Simulate a few random steps
        steps = 0
        total_reward = 0
        
        while not env.done and steps < min(5, task_config.max_steps):
            # Simple random action
            import random
            if observation.stations and random.random() > 0.3:
                station = random.choice(observation.stations)
                action = Action(type=ActionType.SELECT_STATION, station_id=station.id)
            else:
                action = Action(type=ActionType.WAIT, wait_time_minutes=random.randint(5, 15))
            
            obs, reward, done, info = env.step(action)
            total_reward += reward.value
            steps += 1
            
            if done:
                break
        
        final_state = env.state()
        print(f"   Result: Score={final_state.score:.3f}, Steps={steps}, Total Reward={total_reward:.3f}")


if __name__ == "__main__":
    try:
        # Run easy task demo
        demo_easy_task()
        
        # Run all difficulties demo
        demo_all_difficulties()
        
        print("\n\n🎉 Demo completed successfully!")
        print("The EV Charging Environment is working perfectly!")
        print("\nTo run with AI inference:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python inference.py easy")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
