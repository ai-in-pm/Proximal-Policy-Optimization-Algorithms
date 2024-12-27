# Proximal Policy Optimization (PPO) Implementation

A comprehensive implementation of Proximal Policy Optimization (PPO) algorithms in PyTorch, featuring both theoretical foundations and practical demonstrations.

## 🌟 Features

- Clean, modular PyTorch implementation of PPO
- Support for continuous and discrete action spaces
- Implementations of key PPO components:
  - Clipped surrogate objective
  - Value function estimation
  - Generalized Advantage Estimation (GAE)
  - Policy and value function updates
- Multiple environment demonstrations:
  - CartPole-v1
  - LunarLander-v2
- Real-time visualization of agent performance
- Training progress tracking and plotting

## 📋 Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/ai-in-pm/Proximal-Policy-Optimization-Algorithms.git
cd Proximal-Policy-Optimization-Algorithms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run a demo:
```bash
# Run CartPole demo
python demonstrations/cartpole_demo.py

# Run LunarLander demo
python demonstrations/lunar_lander_demo.py
```

## 🏗️ Project Structure

```
.
├── ppo.py              # Core PPO implementation
├── demonstrations/     # Example implementations
│   ├── cartpole_demo.py
│   ├── lunar_lander_demo.py
│   └── README.md
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## 💻 Implementation Details

### Core Components

1. **Actor-Critic Architecture**
   - Actor (Policy) network outputs action distributions
   - Critic (Value) network estimates state values

2. **PPO Algorithm**
   - Clipped surrogate objective for stable updates
   - Value function loss with clipping
   - Entropy bonus for exploration
   - Generalized Advantage Estimation (GAE)

3. **Key Features**
   - Modular design for easy extension
   - Configurable hyperparameters
   - Support for different environments
   - Training progress visualization

### Hyperparameters

- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- GAE parameter (lambda): 0.95
- Clipping parameter (epsilon): 0.2
- Value function coefficient: 1.0
- Entropy coefficient: 0.01

## 📊 Results

The implementation has been tested on various environments:

1. **CartPole-v1**
   - Achieves optimal performance (500 steps) within 500 episodes
   - Stable learning across different random seeds

2. **LunarLander-v2**
   - Achieves landing within 1000 episodes
   - Demonstrates stable control and smooth landing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
2. Gymnasium Documentation: https://gymnasium.farama.org/
3. PyTorch Documentation: https://pytorch.org/docs/
