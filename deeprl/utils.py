import torch
from torch import nn
import numpy as np


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

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

def get_model_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size 