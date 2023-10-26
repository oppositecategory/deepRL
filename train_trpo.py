import gymnasium as gym
from gymnasium.spaces.box import Box

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian
import numpy as np 
from tqdm import tqdm

from torch_rl.trpo import trpo_update 
from torch_rl.models.discrete_policy import DiscretePolicy
from torch_rl.models.continuous_policy import GaussianPolicy

import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.use('tkagg')

import os 

# env_name = "CartPole-v1"
env_name = "BipedalWalker-v3"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]

if isinstance(env.action_space,Box):
    action_space = env.action_space.shape[0]
else:
    action_space = env.action_space.n



batch_size= 5000    
num_epochs = 100

gamma = 0.99
tau = 0.97
KL_bound = 0.01
backtrack_coeff = 0.8
l2_value = 1e-3

policy = DiscretePolicy(input_dim=state_dim,
                        output_dim=action_space,
                        KL_bound = KL_bound,
                        backtrack_coeff = backtrack_coeff)
policy = GaussianPolicy(input_dim=state_dim,
                          output_dim=action_space,
                          KL_bound = KL_bound,
                          backtrack_coeff = backtrack_coeff)

value_net = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
)
policy.load_state_dict(torch.load('policy_BipedalWalker-v3.pt'))
value_net.load_state_dict(torch.load('value_BipedalWalker-v3.pt'))

value_optimizer = Adam(value_net.parameters(), lr=l2_value)


def plot_results(file_name):
    returns = np.load(file_name)

    plt.title("Return value averaged over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Return value")
    plt.grid()


    plt.plot(range(num_epochs),returns)
    plt.show()


def test_trpo():
    observations = [np.random.rand(state_dim) for i in range(50)]
    actions = [np.random.randint(0,action_space,size=(1,)) for i in range(50)]
    rewards = [np.random.normal(loc=torch.tensor(0.), scale=0.1) for i in range(50)]
    returns = list(reward_to_go(rewards))
    mask = [1 if i < 49 else 0 for i in range(50)]

    trpo_update(policy,value_net,value_optimizer, observations,actions, rewards, mask,gamma,tau)


def process_input(x):
    if len(x.shape) == 3:
        return torch.as_tensor(x,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    return torch.as_tensor(x,dtype=torch.float32).unsqueeze(0)
    
def train_epoch():
    obs, info = env.reset()
    observations, actions, rewards = [],[],[]
    episode_count = 0
    mask = []
    num_episodes,reward_epoch = 0,0
    reward_episode = []
    for frame in tqdm(range(batch_size)):
        observations.append(obs.copy())
        action = policy.sample_action(process_input(obs))
        values = value_net(process_input(obs))

        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        reward_episode.append(reward)
        if terminated or truncated or (len(reward_episode) >= 1600 and sum(reward_episode) < 300):
            obs, info = env.reset()
            mask.append(0)
            reward_epoch += sum(reward_episode)
            num_episodes+=1
            reward_episode = []
        else:
            mask.append(1)

    returns = trpo_update(policy,value_net,value_optimizer, observations,actions, rewards, mask,gamma,tau)
    print("Num of episodes carried out:", num_episodes)
    return reward_epoch/num_episodes
 
def train(save=False):
    returns = []
    for i in range(num_epochs):
        average_return = train_epoch()
        print(f"{i}/{num_epochs} Averaged return in batch: {average_return}")
        returns.append(average_return)

    if save:
        torch.save(policy.state_dict(),f'policy_{env_name}.pt')
        torch.save(value_net.state_dict(),f'value_{env_name}.pt')
    returns = np.array(returns)
    np.save(f'{env_name}_experiment',returns)


def test_model():
    obs, info = env.reset()
    terminated = False 
    while not terminated:
        env.render() 
        action = policy.sample_action(process_input(obs))
        # action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)


# train(True)
# plot_results('BipedalWalker-v3_experiment.npy')
# test_model()