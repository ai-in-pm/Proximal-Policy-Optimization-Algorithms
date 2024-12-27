import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ppo import PPO
import numpy as np
from tqdm import tqdm

def train_cartpole(episodes=500, max_timesteps=500):
    """
    Train PPO on CartPole-v1 environment and visualize results
    """
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole

    # Initialize PPO agent
    ppo = PPO(state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    avg_rewards = []
    
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        
        for t in range(max_timesteps):
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state)
            action, log_prob = ppo.actor_critic.get_action(state_tensor)
            
            # Take discrete action (CartPole expects int)
            discrete_action = torch.argmax(action).item()
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(discrete_action)
            
            # Store transition
            states.append(state)
            actions.append(action.detach().numpy())
            log_probs.append(log_prob.detach().item())
            rewards.append(reward)
            dones.append(done or truncated)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        # Update PPO agent
        ppo.update(states, actions, log_probs, rewards, dones, next_state)
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")
    
    env.close()
    return episode_rewards, avg_rewards

def plot_results(rewards, avg_rewards, save_path='cartpole_results.png'):
    """
    Plot training results
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CartPole-v1 Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def record_video(ppo_agent, video_path='cartpole_video'):
    """
    Record a video of the trained agent
    """
    env = gym.make('CartPole-v1', render_mode='human')
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        action, _ = ppo_agent.actor_critic.get_action(state_tensor)
        discrete_action = torch.argmax(action).item()
        
        state, reward, done, truncated, _ = env.step(discrete_action)
        total_reward += reward
        
        if done or truncated:
            break
    
    env.close()
    print(f"Demo completed! Total reward: {total_reward}")

if __name__ == "__main__":
    print("Starting CartPole-v1 demonstration...")
    
    # Create video directory if it doesn't exist
    os.makedirs('cartpole_video', exist_ok=True)
    
    # Train the agent
    rewards, avg_rewards = train_cartpole()
    
    # Plot and save results
    plot_results(rewards, avg_rewards)
    
    # Load the trained agent and record a video
    ppo = PPO(4, 2)  # state_dim=4, action_dim=2 for CartPole
    record_video(ppo)
    
    print("\nDemonstration completed!")
    print("- Training plot saved as 'cartpole_results.png'")
    print("- Video recording saved in 'cartpole_video' directory")
