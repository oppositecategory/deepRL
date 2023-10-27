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

env_name = 'Hopper-v4'
# env_name = 'Humanoid-v4'
# render_mode = None

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]

if isinstance(env.action_space,Box):
    action_space = env.action_space.shape[0]
else:
    action_space = env.action_space.n

batch_size = 5000 
num_epochs = 200

gamma = 0.99
tau = 0.97
KL_bound = 0.01
backtrack_coeff = 0.5
l2_value = 1e-3


# policy = DiscretePolicy(input_dim=state_dim,
#                         output_dim=action_space,
#                         KL_bound = KL_bound,
#                         backtrack_coeff = backtrack_coeff)
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

if f'policy_{env_name}.pt' in os.listdir('weights'):
    print("Detected previous weights...")
    policy.load_state_dict(torch.load(f'weights/policy_{env_name}.pt'))
    value_net.load_state_dict(torch.load(f'weights/value_{env_name}.pt'))

value_optimizer = Adam(value_net.parameters(), lr=l2_value)

def plot_results():
    returns = np.load(f'results/{env_name}_experiment.npy')

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
        if terminated or truncated:
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
        torch.save(policy.state_dict(),f'weights/policy_{env_name}.pt')
        torch.save(value_net.state_dict(),f'weights/value_{env_name}.pt')
    returns = np.array(returns)
    if f'results/{env_name}_experiment' in os.listdir('results'):
        old =  np.load(f'results/{env_name}_experiment.npy')
        results = np.concat([old,results])
    np.save(f'results/{env_name}_experiment',returns)
    plot_results()


def test_model():
    for i in range(10):
        obs, info = env.reset()
        terminated = False 
        while not terminated:
            action = policy.sample_action(process_input(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            print(reward)


train(True)
# test_model()