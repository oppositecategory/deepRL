import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from torch.optim import Adam

import numpy as np 
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

env = gym.make("CarRacing-v2",
              #render_mode='human',
               continuous=False)
env.action_space.seed(42)

# Important hyper-parameters
state_dim = env.observation_space.shape[0]
action_space = env.action_space
epsilon= 0.01
batch_size= 50000
N_buffer= 1000
minibatch_size=32
gamma = 0.99
lr = 1e-2
num_epochs = 10

def conv_block(in_f, out_f, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_f,out_f, kernel_size,stride),
        nn.ReLU()
    )

def conv_output_size(input_dim, filters):
    W = input_dim
    for f in filters:
        W = (W - f[1])/f[2] + 1
    if not W.is_integer():
        raise Exception("The given filter parameters do not match.")
    return int(W)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class QNetwork(nn.Module):
    def __init__(self, 
                 N_buffer=1000, 
                 input_dim=(96,96,1),
                 filters=[(16,8,4),(32,3,2)], 
                 hidden_dim=[256],
                 output_dim=5):
        """
            Implement a Deep-Q Network with the Replay Buffer encapuslated in it aswell.
            Arguments:
                N_buffer: The size of the replay buffer.
                input_dim: The dimension of the input given as (height,width,batch_size),
                filters: Array representing the convolutional filters as: (num_filters, kernel_size, stride)
                hidden_dim: Array representing the number of neurons in each FC layer following the conv layers.
                output_dim: The dimension of the output layer.
            
            NOTE: There is major importance to the dimensions of the input and the conv-net parameters.
                  In the original paper the input was processed into 84x84, however in the CarRace environment 
                  we are given a square input so cropping is redundant, instead I change the size of 
                  the second filter in the original paper to make sure the stride and the dimensions arithmetic add up.
        """
        super().__init__()
        self.N_buffer = N_buffer
        self.memory_index = 0 # Works modulo N_buffer

        self.m_state = torch.zeros(N_buffer,input_dim[0],input_dim[0], dtype=torch.float32)
        self.m_action = torch.zeros(N_buffer,dtype=torch.long)
        self.m_reward = torch.zeros(N_buffer,dtype=torch.float32)
        self.m_state1 = torch.zeros(N_buffer,input_dim[0],input_dim[0],dtype=torch.float32)

        in_channels = [input_dim[2]] + [f[0] for f in filters] # [4,16,32]
        kernels_size = [f[1] for f in filters] # [8,4]
        strides = [f[2] for f in filters] # [4,2]
        
        conv_blocks = [
            conv_block(in_channels[i],in_channels[i+1],kernels_size[i],strides[i])
            for i in range(len(kernels_size))
        ]

        self.conv_net = nn.Sequential(
            *conv_blocks
        )

        self.conv_rep = (conv_output_size(input_dim[0],filters) ** 2) * filters[-1][0]
        hidden_dim = [self.conv_rep] + hidden_dim + [output_dim]
        hidden_layers = [nn.Linear(hidden_dim[i],hidden_dim[i+1]) \
            for i in range(len(hidden_dim)-1)]

        self.fc_layer = nn.Sequential(
            *hidden_layers
        )
    
    def forward(self,x):
        x = self.conv_net(x)
        x = x.view(-1,self.conv_rep)
        x = self.fc_layer(x)
        return x

    def store_experience(self, s):
        # s is given as: (state,action,reward,next_state)
        self.m_state[self.memory_index,:,:] = s[0]
        self.m_action[self.memory_index] = s[1]
        self.m_reward[self.memory_index] = s[2]
        if s[3] != None: 
            self.m_state1[self.memory_index,:,:] = s[3]
        self.memory_index = (self.memory_index + 1) % self.N_buffer

    def sample_experiences(self):
        indices = torch.randint(0, self.N_buffer,size=(minibatch_size,))
        s_samples = self.m_state[indices,:,:]
        s_actions = self.m_action[indices]
        s_rewards = self.m_reward[indices]
        s_samples1 = self.m_state1[indices,:,:]
        return (s_samples,s_actions,s_rewards,s_samples1)

network = QNetwork()
optimizer = Adam(network.parameters(), lr=lr)

@torch.no_grad()
def process_experience_data(batch):
    batch_state, batch_action, batch_reward,batch_next_s = batch
    mask_next_s = torch.count_nonzero(batch_next_s,dim=(-2,-1))
    return_approx = network(batch_next_s[:,None,:,:]).argmax(dim=1)
    X = batch_state 
    y = batch_reward[0] + mask_next_s[...,0] * return_approx
    return X, batch_action, y
    
def train_epoch():
    obs, info = env.reset()

    episode_rewards = []
    rewards = []
    avg_batch_loss = []
    for frame in tqdm(range(batch_size)):
        phi_curr = torch.tensor(rgb2gray(obs)[None,:,:],dtype=torch.float32)

        if np.random.binomial(1,epsilon,1):
            action = env.action_space.sample()
        else:
            action = network(phi_curr).argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()
            episode_rewards.append(np.mean(rewards))
            rewards = []
            phi_next = None
        else:
            phi_next = torch.tensor(rgb2gray(next_obs)[None,:,:],dtype=torch.float32)

        s = (phi_curr, action, reward, phi_next)
        network.store_experience(s)

        if frame < minibatch_size:
            continue
        
        batch = network.sample_experiences()
        X,actions,y = process_experience_data(batch)

        optimizer.zero_grad()
        output = network(X[:,None,:,:])
        output = output[range(minibatch_size),actions.long()]
        batch_loss = ((output - y) ** 2).mean()

        avg_batch_loss.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()
    avg_episode_reward = np.mean(episode_rewards)
    return avg_batch_loss,avg_episode_reward

plt.ion()
# figure, ax = plt.subplots(figsize=(10, 8))
plt.title("Average Q")
plt.xlabel("Training Epochs")
plt.ylabel("Average action value")

plt.axis([0,num_epochs,0,10])


for epoch in range(num_epochs):
    avg_batch_loss, avg_episode_reward = train_epoch()
    plt.plot(epoch,avg_episode_reward)
    plt.pause(0.05)

plt.show()