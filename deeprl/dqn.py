import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np 


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
                 output_dim=5,
                 minibatch_size=32):
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
        self.minibatch_size = minibatch_size

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

    
    @torch.no_grad()
    def sample_and_process_replay(self):
        # Util function to sample mini-batch from the history and process it
        # into a form usable for training the Q-network.
        indices = torch.randint(0, self.N_buffer,size=(self.minibatch_size,))
        batch_state = self.m_state[indices,:,:]
        batch_action = self.m_action[indices]
        batch_reward = self.m_reward[indices]
        batch_next_s = self.m_state1[indices,:,:]

        mask_next_s = torch.count_nonzero(batch_next_s,dim=(-2,-1))
        with torch.no_grad():
            return_approx = self.forward(batch_next_s[:,None,:,:]).argmax(dim=1)
        
        X = batch_state 
        y = batch_reward[0] + mask_next_s[...,0] * return_approx
        return X, batch_action, y
    

  