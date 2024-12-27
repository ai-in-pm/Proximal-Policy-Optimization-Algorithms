import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Actor: Get action mean and std
        action_mean = self.mean_layer(self.actor(state))
        action_std = self.log_std.exp()
        
        # Critic: Get state value
        value = self.critic(state)
        
        return action_mean, action_std, value
    
    def get_action(self, state):
        action_mean, action_std, _ = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        c1=1.0,
        c2=0.01,
        batch_size=64,
        n_epochs=10
    ):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(values)
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        # Convert to tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Get values for all states
        with torch.no_grad():
            _, _, values = self.actor_critic(states)
            _, _, next_value = self.actor_critic(torch.FloatTensor([next_state]))
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values.detach().numpy(), 
                                            next_value.item(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.n_epochs):
            # Get current policy distribution and values
            action_mean, action_std, values = self.actor_critic(states)
            dist = Normal(action_mean, action_std)
            curr_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            
            # Compute ratio and surrogate loss
            ratio = (curr_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            
            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((returns - values.squeeze()) ** 2).mean()
            
            # Total loss
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
            
            # Update networks
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        
    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
