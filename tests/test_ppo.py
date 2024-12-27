import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from ppo import PPO, ActorCritic

def test_actor_critic_initialization():
    state_dim = 4
    action_dim = 2
    actor_critic = ActorCritic(state_dim, action_dim)
    
    # Test network architecture
    assert isinstance(actor_critic, torch.nn.Module)
    assert hasattr(actor_critic, 'actor')
    assert hasattr(actor_critic, 'critic')
    assert hasattr(actor_critic, 'mean_layer')
    assert hasattr(actor_critic, 'log_std')

def test_ppo_initialization():
    state_dim = 4
    action_dim = 2
    ppo = PPO(state_dim, action_dim)
    
    # Test PPO attributes
    assert hasattr(ppo, 'actor_critic')
    assert hasattr(ppo, 'optimizer')
    assert isinstance(ppo.gamma, float)
    assert isinstance(ppo.epsilon, float)
    assert isinstance(ppo.c1, float)
    assert isinstance(ppo.c2, float)

def test_get_action():
    state_dim = 4
    action_dim = 2
    actor_critic = ActorCritic(state_dim, action_dim)
    
    # Test action generation
    state = torch.randn(state_dim)
    action, log_prob = actor_critic.get_action(state)
    
    assert isinstance(action, torch.Tensor)
    assert isinstance(log_prob, torch.Tensor)
    assert action.shape == torch.Size([action_dim])
    assert log_prob.shape == torch.Size([])

def test_compute_gae():
    state_dim = 4
    action_dim = 2
    ppo = PPO(state_dim, action_dim)
    
    # Test GAE computation
    rewards = [1.0, 0.5, 0.8]
    values = [0.9, 0.4, 0.7]
    next_value = 0.6
    dones = [False, False, True]
    
    advantages, returns = ppo.compute_gae(rewards, values, next_value, dones)
    
    assert isinstance(advantages, torch.Tensor)
    assert isinstance(returns, torch.Tensor)
    assert len(advantages) == len(rewards)
    assert len(returns) == len(rewards)

if __name__ == '__main__':
    pytest.main([__file__])
