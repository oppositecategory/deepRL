import torch
import torch.nn as nn
from torch.optim import Adam
import gym
from gym.spaces import Discrete, Box

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

from torch_rl.policy_gradient import policy_gradient_update
from torch_rl.utils import reward_to_go

#NOTE: This is written using the old gym repo and not the updated gymnaisum.

env = gym.make('CartPole-v1')
env.action_space.seed(42)

state_dim = env.observation_space.shape[0]
hidden_dim = 64
action_space = env.action_space.n
lr = 1e-2
batch_size = 5000 # Num of rollouts/trajectories
epochs = 50

network = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim,action_space)
)
optimizer = Adam(network.parameters(), lr=lr)

def policy_train_epoch():
    obs, _ = env.reset()
    terminated = False
    finished_rendering_this_epoch = False
    
    observations,actions = [],[]
    episode_rewards  = [] 
    returns = [] 
    batch_returns,batch_lens= [],[]
    while True:
        if not finished_rendering_this_epoch:
            env.render()
        
        observations.append(obs.copy())

        logits_net = network(
            torch.as_tensor(obs,dtype=torch.float32)
        )
        action_dist = Categorical(logits=logits_net)
        action = action_dist.sample().item()

        obs, reward, terminated, truncated = env.step(action)
        
        actions.append(action)
        episode_rewards.append(reward)

        if terminated or truncated:
            episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
            batch_returns.append(episode_return)
            batch_lens.append(episode_length)

            # returns += [episode_return] * episode_length
            returns += list(reward_to_go(episode_rewards))
            

            obs, terminated, episode_rewards = env.reset(), False, []
            finished_rendering_this_epoch = True

            if len(observations) > batch_size:
                break

    batch_loss, batch_returns, batch_lens = policy_gradient_update(network, 
                                                                   optimizer,
                                                                   observations, 
                                                                   actions, 
                                                                   returns)
    return batch_loss, batch_returns,batch_lens

plt.ion()
plt.title("Average return value over epoch")
plt.xlabel("Epoch")
plt.ylabel("Return value")

plt.axis([0,50,0,200])


for i in range(epochs):
    batch_loss, epoch_return, episode_length = policy_train_epoch()
    plt.scatter(i, np.mean(epoch_return),c='black')
    print(
        'epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
        (i, batch_loss, np.mean(epoch_return), np.mean(episode_length))
    )
    plt.pause(0.05)

plt.show()




        
