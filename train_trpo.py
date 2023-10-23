import gymnasium as gym

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian
import numpy as np 
from tqdm import tqdm

from torch_rl.trpo import * 


env = gym.make("CarRacing-v2",  
               continuous=False)

state_dim = env.observation_space.shape
action_space = env.action_space.n
batch_size= 2000
num_epochs = 50

gamma = 0.99
tau = 0.97
KL_bound = 0.01
backtrack_coeff = 0.8
damping = 1 
l2_value = 1e-3

policy = Policy(input_dim=state_dim,
                output_dim=action_space)

value_net = ConvNet(input_dim=state_dim,
                    output_dim=1,
                    hidden_dim=64)
value_optimizer = Adam(value_net.parameters(), lr=l2_value)

def test_trpo():
    observations = [np.random.rand(96,96,3) for i in range(50)]
    actions = [np.random.randint(0,action_space,size=(1,)) for i in range(50)]
    rewards = [np.random.normal(loc=torch.tensor(0.), scale=0.1) for i in range(50)]
    returns = list(reward_to_go(rewards))
    mask = [1 if i < 49 else 0 for i in range(50)]

    trpo_update(policy,value_net,observations,actions, rewards, mask,gamma,tau)
    # value_loss = update_value_network(value_net,value_optimizer, observations,returns)

def train_epoch():
    obs, info = env.reset()
    observations, actions, rewards = [],[],[]
    episode_rewards = []
    returns = []
    episode_count = 0
    mask = []
    episode_rewards = []
    episode_returns = []
    for frame in tqdm(range(batch_size)):
        observations.append(obs.copy())
        action = policy.sample_action(torch.as_tensor(obs,dtype=torch.float32).permute(2,0,1).unsqueeze(0))
        values = value_net(torch.as_tensor(obs,dtype=torch.float32).permute(2,0,1).unsqueeze(0))

        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        episode_rewards.append(reward)
        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()
            mask.append(0)
        else:
            mask.append(1)

    returns = trpo_update(policy,value_net,observations,actions, rewards, mask,gamma,tau)
    update_value_network(value_net,value_optimizer, observations,returns)
    return torch.mean(returns).item()
 

returns = []
for i in range(num_epochs):
    average_return = train_epoch()
    print(f"Averaged return in batch: {average_return}")
    returns.append(average_return)

torch.save(policy.state_dict(),'policy.pt')
returns = np.array(returns)
np.save('results',returns)