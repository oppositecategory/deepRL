import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np 

from .utils import *

class QNetwork(ConvNet):
    def __init__(self, 
                 N_buffer=1000, 
                 input_dim=(96,96,1),
                 filters=[(16,8,4),(32,3,2)], 
                 hidden_dim=256,
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
        super().__init__(input_dim,output_dim, filters,hidden_dim)
        self.N_buffer = N_buffer
        self.memory_index = 0 # Works modulo N_buffer
        self.minibatch_size = minibatch_size

        self.m_state = torch.zeros(N_buffer,input_dim[0],input_dim[0], dtype=torch.float32)
        self.m_action = torch.zeros(N_buffer,dtype=torch.long)
        self.m_reward = torch.zeros(N_buffer,dtype=torch.float32)
        self.m_state1 = torch.zeros(N_buffer,input_dim[0],input_dim[0],dtype=torch.float32)
    
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