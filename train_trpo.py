import gymnasium as gym

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian

import numpy as np 
from tqdm import tqdm

from deeprl.trpo import * 


env = gym.make("CarRacing-v2",  
               continuous=False)

state_dim = env.observation_space.shape
action_space = env.action_space.n
batch_size= 2000
num_epochs = 10
gamma = 0.9954
KL_bound = 1e-2
backtrack_coeff = 0.8
damping = 1 

value_net = nn.Sequential(
                nn.Linear(np.prod(state_dim), 256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,1)
)
policy = Policy(input_dim=state_dim,
                output_dim=action_space)


def generate_testing_data():
    observations = [np.random.rand(96,96,3) for i in range(50)]
    actions = [np.random.randint(0,action_space,size=(1,)) for i in range(50)]
    rewards = [np.random.normal(loc=torch.tensor(0.), scale=0.1) for i in range(50)]
    returns = list(reward_to_go(rewards))
    mask = [1 if i < 49 else 0 for i in range(50)]
    return observations, action_space,rewards, returns, mask

def train_epoch():
    obs, info = env.reset()
    observations, actions, rewards = [],[],[]
    episode_rewards = []
    returns = []
    episode_count = 0
    mask = []
    for frame in tqdm(range(batch_size)):
        observations.append(obs.copy())
        action = policy.sample_action(torch.as_tensor(obs,dtype=torch.float32).permute(2,0,1).unsqueeze(0))
        values = value_net(torch.as_tensor(obs,dtype=torch.float32).view(-1))

        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()
            returns += list(reward_to_go(rewards))
            episode_rewards.append(np.sum(rewards))
            mask.append(0)
            rewards = []
        else:
            mask.append(1)

    trpo_update(policy,value_net,observations,actions, returns, mask,gamma)
 
train_epoch()