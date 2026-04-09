"""
Inference script for EV Charging Environment with OpenAI API integration.

This script uses the OpenAI API to make decisions in the EV charging environment
and follows the strict logging format required by OpenEnv.
"""

import os
import json
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
import openai
from pydantic import ValidationError

from models import Observation, Action, ActionType, TaskConfig
from environment import EVChargingEnvironment
from tasks import get_task_config, grade_task

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)


class EVChargingAgent:
    """
    AI agent for EV charging station selection using OpenAI API.
    """
    
    def __init__(self):
        """
        Initialize the agent using environment variables.
        """
        # Prioritize platform-provided API_KEY and API_BASE_URL
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("API_BASE_URL")
        model = os.getenv("MODEL_NAME", "gpt-4")
        
        self.api_key_available = True
        if not api_key:
            logger.warning("No API key found in API_KEY or OPENAI_API_KEY. Agent will use fallback heuristic decisions.")
            self.api_key_available = False
            api_key = "dummy-key-for-validation"
        
        # Configure OpenAI client
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI agent."""
        return """You are an intelligent electric vehicle (EV) charging assistant. Your task is to help EV drivers select the optimal charging station based on multiple factors.

Your goal is to balance:
1. Distance to the charging station
2. Charging cost per kWh
3. Station availability and waiting time
4. Time constraints
5. Budget limitations

You must respond with a JSON object containing your action decision. The action should be one of:
- select_station: Choose a specific charging station
- wait: Wait for a station to become available
- move_to_next_station: Move to the next available station

Response format:
{
    "reasoning": "Brief explanation of your decision",
    "action": {
        "type": "select_station|wait|move_to_next_station",
        "station_id": "station_X" (only for select_station),
        "wait_time_minutes": 15 (only for wait)
    }
}

Consider the EV's current battery level, priority, and the competitive environment with other EVs. Make strategic decisions that optimize for the specific task difficulty."""
    
    def _format_observation_for_prompt(self, obs: Observation) -> str:
        """Format observation for the AI prompt."""
        stations_info = []
        for station in obs.stations:
            stations_info.append(f"""
Station {station.id}:
- Name: {station.name}
- Location: ({station.latitude:.4f}, {station.longitude:.4f})
- Power: {station.power_kw} kW
- Price: ${station.price_per_kwh:.2f}/kWh
- Status: {station.status}
- Waiting time: {station.waiting_time_minutes} minutes
- Queue: {station.current_queue_length}/{station.max_queue_length}
""")
        
        other_evs_info = []
        for ev in obs.other_evs_waiting:
            other_evs_info.append(f"""
EV {ev['id']} at {ev['station_id']}:
- Priority: {ev['priority']}
- Battery: {ev['battery_percent']:.1f}%
- Waiting for: {ev['arrival_time']} minutes
""")
        
        prompt = f"""
Current Situation:
- Your EV: {obs.ev.id}
- Battery: {obs.ev.current_battery_percent:.1f}% ({obs.ev.battery_capacity_kwh} kWh capacity)
- Priority: {obs.ev.priority}
- Consumption: {obs.ev.consumption_rate_kwh_per_km} kWh/km
- Current location: ({obs.current_location_lat:.4f}, {obs.current_location_lon:.4f})
- Destination: ({obs.destination_lat:.4f}, {obs.destination_lon:.4f})
- Time remaining: {obs.time_remaining_hours:.2f} hours
- Budget remaining: ${obs.budget_remaining:.2f}
- Step: {obs.step_count}/{obs.max_steps}

Available Charging Stations:
{''.join(stations_info)}

Other EVs Waiting:
{''.join(other_evs_info)}

Make the optimal decision based on this information.
"""
        return prompt
    
    def decide_action(self, observation: Observation) -> Action:
        """
        Decide on an action based on the current observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Selected action
        """
        if not self.api_key_available:
            # Fallback heuristic: Select the nearest available station or wait
            available_stations = [s for s in observation.stations if s.status == 'available']
            if available_stations:
                # Simple distance-based selection
                best_station = min(available_stations, key=lambda s: abs(s.latitude - observation.current_location_lat) + abs(s.longitude - observation.current_location_lon))
                logger.info(f"Fallback heuristic: Selecting nearest station {best_station.id}")
                return Action(type=ActionType.SELECT_STATION, station_id=best_station.id)
            
            logger.info("Fallback heuristic: No available stations, waiting.")
            return Action(type=ActionType.WAIT, wait_time_minutes=10)

        try:
            # Format observation for prompt
            prompt = self._format_observation_for_prompt(observation)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.info(f"AI Response: {content}")
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = content[start_idx:end_idx]
                decision = json.loads(json_str)
                
                # Create action from decision
                action_data = decision.get("action", {})
                action_type = ActionType(action_data.get("type", "wait"))
                
                action = Action(
                    type=action_type,
                    station_id=action_data.get("station_id"),
                    wait_time_minutes=action_data.get("wait_time_minutes")
                )
                
                logger.info(f"Decided action: {action.dict()}")
                return action
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse AI response: {e}")
                # Fallback to wait action
                return Action(type=ActionType.WAIT, wait_time_minutes=10)
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fallback to wait action
            return Action(type=ActionType.WAIT, wait_time_minutes=10)


class InferenceRunner:
    """
    Runner for inference with proper logging format.
    """
    
    def __init__(self, difficulty: str):
        """
        Initialize the inference runner.
        
        Args:
            difficulty: Task difficulty level
        """
        self.difficulty = difficulty
        self.agent = EVChargingAgent()
        self.task_config = get_task_config(difficulty)
        self.environment = EVChargingEnvironment(self.task_config)
    
    def run_episode(self, seed: int = 42, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete episode with logging.
        
        Args:
            seed: Random seed
            max_steps: Maximum steps (overrides task config if provided)
            
        Returns:
            Episode results
        """
        print("[START]")
        
        try:
            # Reset environment
            observation = self.environment.reset()
            
            episode_data = {
                "difficulty": self.difficulty,
                "seed": seed,
                "steps": [],
                "total_reward": 0.0,
                "final_score": 0.0,
                "done": False,
                "error": None
            }
            
            max_steps = max_steps or self.task_config.max_steps
            
            for step in range(max_steps):
                print(f"[STEP] {step + 1}")
                
                try:
                    # Agent decision
                    action = self.agent.decide_action(observation)
                    
                    # Environment step
                    next_observation, reward, done, info = self.environment.step(action)
                    
                    # Record step
                    step_data = {
                        "step": step + 1,
                        "observation": observation.dict(),
                        "action": action.dict(),
                        "reward": reward.dict(),
                        "done": done,
                        "info": info
                    }
                    episode_data["steps"].append(step_data)
                    episode_data["total_reward"] += reward.value
                    
                    observation = next_observation
                    
                    if done:
                        episode_data["done"] = True
                        break
                        
                except Exception as e:
                    episode_data["error"] = str(e)
                    break
            
            # Get final state and grade
            final_state = self.environment.state()
            episode_data["final_score"] = final_state.score
            episode_data["grade"] = grade_task(final_state)
            
            print("[END]")
            
            return episode_data
            
        except Exception as e:
            print("[END]")
            episode_data["error"] = str(e)
            return episode_data
    
    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """
        Save inference results to a JSON file.
        Appends to existing results if file exists to support multiple tasks.
        
        Args:
            results: Results dictionary to save
            output_file: Output file path (auto-generated if None)
        """
        if output_file is None:
            output_file = "submission.json"
        
        existing_data = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except Exception as e:
                logger.warning(f"Could not load existing {output_file}: {e}")
                existing_data = []
        
        # Add new results
        existing_data.append(results)
        
        # Save as a list of task results
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file} (Total tasks recorded: {len(existing_data)})")
        print(f"Results saved to {output_file}")


def main():
    """
    Main entry point for inference.
    
    Usage:
        python inference.py <difficulty> [--seed SEED] [--output FILE] [--max-steps STEPS]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EV charging inference")
    parser.add_argument("difficulty", choices=["easy", "medium", "hard"], 
                       nargs='?', default=os.getenv("TASK_DIFFICULTY", "easy"),
                       help="Task difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--max-steps", type=int, help="Maximum steps (overrides task config)")
    
    args = parser.parse_args()
    
    # Log configuration
    logger.info(f"Starting inference with difficulty: {args.difficulty}")
    logger.info(f"Seed: {args.seed}")
    if args.output:
        logger.info(f"Output file: {args.output}")
    if args.max_steps:
        logger.info(f"Max steps: {args.max_steps}")
    
    try:
        # Create runner
        runner = InferenceRunner(difficulty=args.difficulty)
        
        # Run episode
        results = runner.run_episode(
            seed=args.seed,
            max_steps=args.max_steps
        )
        
        # Save results
        runner.save_results(results, args.output)
        
        # Print summary
        print(f"\nInference Summary:")
        print(f"Difficulty: {results['difficulty']}")
        print(f"Seed: {results['seed']}")
        print(f"Final Score: {results['final_score']:.4f}")
        print(f"Total Reward: {results['total_reward']:.4f}")
        print(f"Steps Taken: {len(results['steps'])}")
        print(f"Episode Done: {results['done']}")
        
        if results.get('error'):
            print(f"Error: {results['error']}")
        
    except Exception as e:
        import traceback
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
