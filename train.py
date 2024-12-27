import gymnasium as gym
import numpy as np
import torch
from ppo import PPO
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_ppo(env_name="Pendulum-v1", max_episodes=1000, max_timesteps=200):
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize PPO agent
    ppo = PPO(state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    
    for episode in tqdm(range(max_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        
        for t in range(max_timesteps):
            # Get action from policy
            state_tensor = torch.FloatTensor(state)
            action, log_prob = ppo.actor_critic.get_action(state_tensor)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action.detach().numpy())
            
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
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards

def plot_results(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_progress.png')
    plt.close()

if __name__ == "__main__":
    # Train the agent
    rewards = train_ppo()
    
    # Plot and save results
    plot_results(rewards)
    
    print("Training completed! Results saved as 'training_progress.png'")
