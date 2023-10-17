import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
from gym.spaces import Discrete, Box

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

"""
NOTE: This is written using the old gym repo and not the updated gymnaisum.
"""

env = gym.make('CartPole-v0')
env.action_space.seed(42)

state_dim = env.observation_space.shape[0]
hidden_dim = 64
action_space = env.action_space.n
lr = 1e-2
batch_size = 5000
epochs = 50

network = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim,action_space)
)
optimizer = Adam(network.parameters(), lr=lr)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def policy_train_epoch():
    """
        The function carries out an epoch of training; a batch here means a number of rollouts/episodes 
        where it's rollout length is arbitrary. 

        The gradient is calculated at the end of the function but with the data gathered over all the episodes
        carried out.
    """
    obs = env.reset()
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

    optimizer.zero_grad()
    # NOTE: Because we need the backwards() to calculate
    # the gradient of the logits with respect to the network's 
    # parameters, we need to re-evaluate the network at the captured observations as otherwise they are detached.
    obs_tensor = torch.tensor(observations,
                              dtype=torch.float32)
    actions_tensor = torch.tensor(actions,
                                  dtype=torch.int32)
    returns = torch.tensor(returns,
                           dtype=torch.float32)

    logits = network(obs_tensor)
    actions_distribution = Categorical(logits=logits)
    log_probs = actions_distribution.log_prob(actions_tensor)
    batch_loss = -(log_probs * returns).mean()

    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_returns,batch_lens

plt.ion()
# figure, ax = plt.subplots(figsize=(10, 8))
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




        
