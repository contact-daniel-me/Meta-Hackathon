#!/usr/bin/env python3
"""
Simple test script for EV Charging Environment.
"""

from tasks import get_task_config, run_task_episode
from environment import EVChargingEnvironment
from models import Action, ActionType


def test_all_tasks():
    """Test all three difficulty levels."""
    print("Testing EV Charging Environment...")
    print("=" * 50)
    
    # Test easy task
    print("\n1. Testing Easy Task...")
    try:
        results = run_task_episode('easy', seed=42)
        print(f"   ✓ Easy task completed successfully")
        print(f"   Grade: {results['grade']:.3f}")
        print(f"   Steps: {results['steps']}")
        print(f"   Final Battery: {results['final_battery']:.1f}%")
    except Exception as e:
        print(f"   ✗ Easy task failed: {e}")
        return False
    
    # Test medium task
    print("\n2. Testing Medium Task...")
    try:
        results = run_task_episode('medium', seed=42)
        print(f"   ✓ Medium task completed successfully")
        print(f"   Grade: {results['grade']:.3f}")
        print(f"   Steps: {results['steps']}")
        print(f"   Final Battery: {results['final_battery']:.1f}%")
    except Exception as e:
        print(f"   ✗ Medium task failed: {e}")
        return False
    
    # Test hard task
    print("\n3. Testing Hard Task...")
    try:
        results = run_task_episode('hard', seed=42)
        print(f"   ✓ Hard task completed successfully")
        print(f"   Grade: {results['grade']:.3f}")
        print(f"   Steps: {results['steps']}")
        print(f"   Final Battery: {results['final_battery']:.1f}%")
    except Exception as e:
        print(f"   ✗ Hard task failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    return True


def test_manual_environment():
    """Test manual environment interaction."""
    print("\nTesting Manual Environment Interaction...")
    print("-" * 40)
    
    try:
        # Create environment
        task_config = get_task_config('easy')
        env = EVChargingEnvironment(task_config, seed=123)
        
        # Reset environment
        observation = env.reset()
        print(f"Environment reset successfully")
        print(f"EV Battery: {observation.ev.current_battery_percent:.1f}%")
        print(f"Available stations: {len(observation.stations)}")
        
        # Take a simple action
        if observation.stations:
            first_station = observation.stations[0]
            action = Action(
                type=ActionType.SELECT_STATION,
                station_id=first_station.id
            )
            
            obs, reward, done, info = env.step(action)
            print(f"Action completed: {action.type.value}")
            print(f"Reward: {reward.value:.4f}")
            print(f"Done: {done}")
        
        print("✓ Manual environment test passed")
        return True
        
    except Exception as e:
        print(f"✗ Manual environment test failed: {e}")
        return False


if __name__ == "__main__":
    success = True
    
    # Test all tasks
    if not test_all_tasks():
        success = False
    
    # Test manual interaction
    if not test_manual_environment():
        success = False
    
    if success:
        print("\n🎉 All environment tests completed successfully!")
        print("The EV Charging Environment is ready to use!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
