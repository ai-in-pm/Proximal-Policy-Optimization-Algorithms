import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ppo import PPO
import numpy as np
from tqdm import tqdm

def train_lunar_lander(episodes=1000, max_timesteps=1000):
    """
    Train PPO on LunarLander-v2 environment
    """
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]  # 8 for LunarLander
    action_dim = env.action_space.n  # 4 for LunarLander

    # Initialize PPO agent with modified hyperparameters for LunarLander
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        c1=0.5,  # Reduced value loss coefficient
        c2=0.01  # Entropy coefficient
    )
    
    # Training metrics
    episode_rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        
        for t in range(max_timesteps):
            state_tensor = torch.FloatTensor(state)
            action, log_prob = ppo.actor_critic.get_action(state_tensor)
            
            # Take discrete action
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
        
        # Track progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        # Save best model
        if avg_reward > best_avg_reward and episode > 100:
            best_avg_reward = avg_reward
            torch.save(ppo.actor_critic.state_dict(), 'lunar_lander_best.pt')
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")
    
    env.close()
    return episode_rewards, avg_rewards

def plot_results(rewards, avg_rewards, save_path='lunar_lander_results.png'):
    """
    Plot training results
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('LunarLander-v2 Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def record_video(ppo_agent, video_path='lunar_lander_video'):
    """
    Record a video of the trained agent
    """
    env = gym.make('LunarLander-v2', render_mode='human')
    
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
    print("Starting LunarLander-v2 demonstration...")
    
    # Create video directory if it doesn't exist
    os.makedirs('lunar_lander_video', exist_ok=True)
    
    # Train the agent
    rewards, avg_rewards = train_lunar_lander()
    
    # Plot and save results
    plot_results(rewards, avg_rewards)
    
    # Load the best model and record a video
    ppo = PPO(8, 4)  # state_dim=8, action_dim=4 for LunarLander
    ppo.actor_critic.load_state_dict(torch.load('lunar_lander_best.pt'))
    record_video(ppo)
    
    print("\nDemonstration completed!")
    print("- Training plot saved as 'lunar_lander_results.png'")
    print("- Best model saved as 'lunar_lander_best.pt'")
    print("- Video recording saved in 'lunar_lander_video' directory")
