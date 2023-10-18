import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import Adam
from deeprl import dqn
import numpy as np 
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

env = gym.make("CarRacing-v2",  
               render_mode='human',
               continuous=False)
env.action_space.seed(42)

# Important hyper-parameters
state_dim = env.observation_space.shape[0]
action_space = 5
epsilon= 0.01 
batch_size= 50000
minibatch_size=32
N_buffer= 10000
gamma = 0.99
lr = 1e-4
num_epochs = 10


network = dqn.QNetwork(N_buffer=N_buffer,
                       minibatch_size=minibatch_size,
                       input_dim=(state_dim,state_dim,1),
                       output_dim=action_space)
optimizer = Adam(network.parameters(), lr=lr)

def train_epoch():
    obs, info = env.reset()

    episode_rewards = []
    rewards = []
    avg_batch_loss = []
    for frame in tqdm(range(batch_size+minibatch_size)):
        phi_curr = torch.tensor(dqn.rgb2gray(obs)[None,:,:],dtype=torch.float32)

        if np.random.binomial(1,epsilon,1):
            action = env.action_space.sample()
        else:
            action = network(phi_curr).argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()
            episode_rewards.append(np.mean(rewards))
            print(f"Episode {len(episode_rewards)}: reward: {np.mean(rewards)}")
            rewards = []
            phi_next = None
        else:
            phi_next = torch.tensor(dqn.rgb2gray(next_obs)[None,:,:],dtype=torch.float32)

        s = (phi_curr, action, reward, phi_next)
        network.store_experience(s)

        if frame < minibatch_size:
            continue
        
        X,actions,y = network.sample_and_process_replay()

        optimizer.zero_grad()
        output = network(X[:,None,:,:])
        output = output[range(minibatch_size),actions.long()]
        batch_loss = ((output - y) ** 2).mean()

        avg_batch_loss.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    avg_episode_reward = np.mean(episode_rewards)
    return avg_batch_loss,avg_episode_reward

train_epoch()
  