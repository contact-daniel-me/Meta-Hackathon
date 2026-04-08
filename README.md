---
title: Ev Openenv
emoji: 🚀
colorFrom: green
colorTo: pink
sdk: docker
app_file: server/app.py
pinned: false
---

# EV Charging Station Optimization Environment

An OpenEnv-compatible environment where AI agents learn to optimize electric vehicle (EV) charging station selection based on distance, cost, availability, and time constraints.

## 🚗 Overview

This environment simulates real-world EV charging scenarios where agents must make strategic decisions about which charging station to use. The environment features dynamic station availability, competing EVs, budget constraints, and time pressure - all factors that EV drivers face in reality.

## 🎯 Features

- **OpenEnv Compatible**: Full compliance with OpenEnv specifications
- **Three Difficulty Levels**: Easy, Medium, and Hard tasks with increasing complexity
- **Realistic Simulation**: Based on actual EV charging dynamics and constraints
- **Deterministic Grading**: Fair and reproducible evaluation system
- **AI Integration**: Built-in OpenAI API support for intelligent agents
- **Docker Ready**: Containerized deployment for Hugging Face Spaces

## 📋 Tasks

### Easy: Nearest Station Selection
**Objective**: Select the closest available charging station to minimize travel distance.

- **Focus**: Proximity optimization
- **Stations**: 5
- **Other EVs**: 2
- **Time Limit**: 4 hours
- **Budget**: $50

### Medium: Cost and Time Optimization
**Objective**: Balance between distance, cost, and waiting time to find the optimal station.

- **Focus**: Multi-factor optimization
- **Stations**: 8
- **Other EVs**: 5
- **Time Limit**: 3 hours
- **Budget**: $40

### Hard: Multi-user Priority Scheduling
**Objective**: Navigate complex scenarios with multiple competing EVs, limited resources, and priority-based scheduling.

- **Focus**: Strategic decision-making under competition
- **Stations**: 6
- **Other EVs**: 10
- **Time Limit**: 2 hours
- **Budget**: $30

## 🏗️ Project Structure

```
ev-charging-environment/
├── models.py          # Pydantic models for Observation, Action, Reward
├── environment.py      # Core environment with OpenEnv interface
├── tasks.py           # Task definitions and deterministic graders
├── grader.py          # Grader implementation for evaluation
├── inference.py       # AI agent with OpenAI integration
├── openenv.yaml       # OpenEnv configuration
├── Dockerfile         # Docker container configuration
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ev-charging-environment
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

#### Running Inference

```bash
# Easy task
python inference.py easy

# Medium task
python inference.py medium --model gpt-4 --seed 123

# Hard task with custom output
python inference.py hard --output results.json --max-steps 25
```

#### Using the Grader

```bash
# Grade a submission
python grader.py medium submission.json
```

#### Programmatic Usage

```python
from environment import EVChargingEnvironment
from tasks import get_task_config
from models import Action, ActionType

# Set up task
task_config = get_task_config("medium")
env = EVChargingEnvironment(task_config, seed=42)

# Reset environment
observation = env.reset()

# Take an action
action = Action(type=ActionType.SELECT_STATION, station_id="station_1")
observation, reward, done, info = env.step(action)

# Get environment state
state = env.state()
print(f"Score: {state.score}")
```

## 🧠 AI Agent Integration

The environment includes a sophisticated AI agent that uses OpenAI's models to make intelligent charging decisions:

### Agent Capabilities

- **Multi-factor Analysis**: Considers distance, cost, availability, and time
- **Strategic Planning**: Adapts to changing conditions and competition
- **Priority Handling**: Makes decisions based on EV priority levels
- **Dynamic Response**: Reacts to station availability changes

### Agent Configuration

```python
from inference import EVChargingAgent

# Initialize agent
agent = EVChargingAgent(
    api_key="your-api-key",
    model="gpt-4"
)

# Make decision
action = agent.decide_action(observation)
```

## 📊 Evaluation and Scoring

### Scoring Components

Each task is evaluated on multiple dimensions:

- **Battery Level**: Final battery percentage (0-30%)
- **Time Efficiency**: Time used vs. time limit (0-20%)
- **Cost Efficiency**: Budget used vs. budget limit (0-20%)
- **Distance Optimization**: Travel distance minimization (0-20%)
- **Decision Quality**: Quality of strategic choices (0-10%)

### Grade Calculation

Final grades are calculated using deterministic grader functions:
- **Easy Grader**: Focuses on proximity and availability
- **Medium Grader**: Balances cost, time, and distance
- **Hard Grader**: Evaluates strategic adaptation and competition handling

## 🐳 Docker Deployment

### Building the Container

```bash
docker build -t ev-charging-env .
```

### Running the Container

```bash
# Local run
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY ev-charging-env

# Hugging Face Spaces compatible
docker run -p 7860:7860 -e OPENAI_API_KEY=$OPENAI_API_KEY ev-charging-env
```

### Hugging Face Spaces

The environment is fully compatible with Hugging Face Spaces:

1. Fork this repository
2. Set `OPENAI_API_KEY` as a repository secret
3. Deploy to Spaces using the provided Dockerfile

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ --cov=. --cov-report=html
```

## 📈 Performance Metrics

The environment tracks comprehensive metrics:

- **Grade**: Final task score (0.0-1.0)
- **Total Reward**: Cumulative reward during episode
- **Steps Taken**: Number of actions taken
- **Final Score**: Environment's internal scoring
- **Episode Completion**: Whether the episode finished successfully

## 🔧 Configuration

### Environment Parameters

```python
task_config = TaskConfig(
    name="Custom Task",
    difficulty="medium",
    max_steps=20,
    num_stations=10,
    num_other_evs=8,
    time_limit_hours=3.5,
    budget_limit=45.0,
    reward_weights={
        "cost_efficiency": 0.3,
        "time_efficiency": 0.3,
        "distance": 0.2,
        "availability": 0.2
    }
)
```

### Reward Weights

Customize reward functions by adjusting weights:

- **Distance**: Penalty for travel distance
- **Cost**: Penalty for charging expenses
- **Time**: Penalty for time consumption
- **Availability**: Reward for finding available stations
- **Efficiency**: Reward for optimal decisions

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd ev-charging-environment

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenEnv team for the environment specification
- OpenAI for providing the AI models
- EV charging industry professionals for domain insights

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the examples in `examples/`

## 🗺️ Roadmap

Future enhancements planned:

- [ ] Additional charging station types (fast, ultra-fast)
- [ ] Weather and traffic integration
- [ ] Multi-agent cooperation scenarios
- [ ] Real-world charging network data integration
- [ ] Mobile app interface for human-in-the-loop testing

---

**Built with ❤️ for the EV charging community**
