import gymnasium as gym

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian

import numpy as np 
from tqdm import tqdm

env = gym.make("CarRacing-v2",  
               continuous=False)

state_dim = env.observation_space.shape[0]
action_space = 5
epsilon= 0.01 
batch_size= 100
gamma = 0.9954
KL_bound = 0.001
backtrack_coeff = 0.8
num_epochs = 10
damping = 1 

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

class Policy(nn.Module):
    def __init__(self, 
                 input_dim=(96,96,3),
                 filters=[(16, 4,2)],   
                 hidden_dim=256,
                 output_dim=action_space):
        super().__init__()
        # Although TRPO itself doesn't have any memory, as a cleaner interface I chose 
        # to hold the batch of rollouts inside the model itself.

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
        self.fc_layer = nn.Sequential(
            nn.Linear(self.conv_rep, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1,self.conv_rep)
        x = self.fc_layer(x)
        return x
    
    def sample_action(self, x):
        logits = self.forward(x)
        action = Categorical(logits=logits).sample().item()
        return action
    
    def log_probs(self, x):
        logits = self.forward(x)
        actions_distribution = Categorical(logits=logits)
        log_probs = actions_distribution.log_prob(actions_tensor)
        return log_probs
    
def get_kl(x):
    """
    NOTE: The KL Divergence is only needed for it's Hessian and it's evaluated at the old policy parameters.
          Observe that action_probs1.data is detaching the gradient from the tensor and hence this function is
          merely used for evaluating the Hessian at the current policy.
    """
    action_probs1 = policy(x)

    action_probs0 = action_probs1.data
    kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
    return kl.sum(1,keepdim=True)

    
value_net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,1)
)
policy = Policy()

@torch.no_grad()
def estimate_advantage(states, rewards, actions, masks):
    # Estimating the advantage using a Value func approximator and TD difference.
    values = value_net(states)
    advantages = torch.Tensor(states.size(0))

    for i in range(states.size(0)-1):
        advantages[i] = rewards[i] + gamma* values[i+1] - masks[i] * values[i]
    return advantages

def conjugate_gradient(Hv,b,N):
    """
        Implementation of the conjugate gradient algorithm.
        Note that Hv which is the matrix we solve for is assumed to be a function to reduce the need to store it.
    """
    threshold = 1e-10
    x = torch.zeros_like(b)
    r = b - Hv(x)
    p = r 
    for i in range(N):
        _Hv = Hv(p)
        alpha = torch.dot(r.T,r) / torch.dot(p.T, _Hv)
        x += alpha*p 
        r_next = r - alpha*_Hv
        if torch.norm(r) < threshold:
            break 
        beta = torch.dot(r_next.T,r_next) / torch.dot(r,r)
        p = r_next + beta*p 
        r = r_next
    return x 

@torch.no_grad()
def backtrack_line_search(f,grad_f,x,p,alpha_0,c=0.1,max=10):
    """ Implements backtrack line search.
        Args:
            - f: the function we wish to optimize
            - grad_f: the gradient of f 
            - x: starting position
            - p: search direction
            - alpha_0: initial step size guess
            - c: control parameter

        This is a bit of a specific variation of the algorithm. Instead of introducting another parameter tau as the shrinkage pace we use exponents of alpha and return the smallest exponent j, such as alpha^j minmize the function as desired.
    """
    m = torch.dot(grad_f,p)
    alpha = alpha_0
    t = -c*m
    sucess = False
    for j in range(max):
        if f(x) - f(x + alpha*p) >= alpha*t:
            sucess = True
            break 
        alpha = alpha*alpha
    return sucess,j

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

def trpo_update(policy, observations, actions, returns, mask):
    actions_t = torch.Tensor(actions)
    rewards_t = torch.Tensor(returns)
    obs_t = torch.Tensor(observations).permute(0,3,1,2)
    mask = torch.Tensor(mask)


    def Hv(v):
        """
        This function implements a neat trick to calculate H@v where H=the Hessian of the averaged KL divergence.
        Considering we are only interested in solution to Hs=g using conjugate gradient, we are basically only
        intersted in the matrix-product Hx and not H itself. Hence it can be shown that the Hessian-vector product 
        is equal to derivative of the product of the first derivative of KL w.r.t to parameters multiplied by the input.
        """
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    advtanges = estimate_advantage(obs_t,rewards_t,actions_t,mask)
    advantages = (advantages - advantages.mean()) / advantages.std() # Normalize advantages

    action_probs = policy.log_probs(obs_t)
    actions_probs_old = actions_probs.data

    # Policy gradient with respect to the loss function.
    loss_fn = (torch.exp(actions_probs - actions_probs_old) * advantages).mean()
    g = torch.autograd.grad(loss_fn, policy.parameters())
    g_vect = torch.utils._flatten_dense_tensors(g)

    # Approximate the Hessian of the KL divergence using 
    direction = conjugate_gradient(Hv, -g_vect,10) # H^(-1)*g approximation.

    # The maximum step size
    denom = direction * Hv(direction)
    step_size = torch.sqrt(torch.sqrt(2*KL_bound)/denom)
    full_step = step_size*direction

    old_params = get_model_params(policy)
    backtrack_line_search(f,grad_f,x,p,alpha_0,c,max=10)
    success,j = backtrack_line_search(policy,
                              g_vect,
                              old_params,
                              full_step,
                              backtrack_coeff)

    if success:
        update = (backtrack_coeff ** j) * full_step
    else:
        update = full_step
    
    print(f"grad_norm: {g_vect.norm()} average return: {returns.mean()}")

    old_params = get_model_params(policy)
    new_params = old_params + optimized_direction
    set_flat_params_to(policy, new_params)

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
        values = value_net(torch.as_tensor(obs,dtype=torch.float32).permute(2,0,1).unsqueeze(0))

        next_obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            episode_count += 1
            returns += list(reward_to_go(episode_rewards))
            mask.append(0)

            obs, terminated, rewards = env.reset(), False, []
            finished_rendering_this_epoch = True
        else:
            mask.append(1)

    trpo_update(policy, observations,actions, returns, mask)

 
train_epoch()