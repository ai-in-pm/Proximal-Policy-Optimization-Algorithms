# PPO Demonstrations

This directory contains demonstrations of the PPO implementation across different environments. Each demonstration includes training, visualization, and video recording of the trained agent.

## Available Demonstrations

### 1. CartPole-v1 (`cartpole_demo.py`)
A classic control problem where a pole is attached to a cart moving along a frictionless track. The goal is to prevent the pole from falling over by moving the cart left or right.

Features:
- Discrete action space (2 actions)
- Simple state space (4 dimensions)
- Quick training (500 episodes)
- Video recording of trained agent

To run:
```bash
python demonstrations/cartpole_demo.py
```

### 2. LunarLander-v2 (`lunar_lander_demo.py`)
A more complex environment where the agent needs to land a spacecraft between two flags. The agent must control the main engine and side thrusters to achieve a safe landing.

Features:
- Discrete action space (4 actions)
- More complex state space (8 dimensions)
- Longer training (1000 episodes)
- Model saving and loading
- Video recording of best performance

To run:
```bash
python demonstrations/lunar_lander_demo.py
```

## Output Files

Each demonstration will generate:
1. Training progress plot (`.png`)
2. Video recording of the trained agent (in respective video directories)
3. Saved model weights (for LunarLander)

## Visualization

The training progress plots show:
- Raw episode rewards (blue, transparent)
- Moving average over 100 episodes (orange)

## Requirements

All demonstrations use the same dependencies as the main project. Make sure you have installed all requirements:
```bash
pip install -r ../requirements.txt
```
