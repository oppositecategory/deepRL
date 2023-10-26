import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian

from .models.continuous_policy import GaussianPolicy
from .models.discrete_policy import DiscretePolicy
from .utils import *

@torch.no_grad()
def estimate_advantage(value_net,states, rewards, actions, masks,gamma,tau):
    values = value_net(states)

    returns = torch.Tensor(rewards.size(0),1)
    deltas = torch.Tensor(rewards.size(0),1)
    advantages = torch.Tensor(rewards.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]
    return returns, advantages

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
        alpha = torch.dot(r,r) / torch.dot(p, _Hv)
        x += alpha*p 
        r_next = r - alpha*_Hv
        if torch.norm(r) < threshold:
            break 
        beta = torch.dot(r_next,r_next) / torch.dot(r,r)
        p = r_next + beta*p 
        r = r_next
    return x 

def trpo_update(policy, value_net,value_optimizer, obs, actions, rewards, mask,gamma,tau):
    if len(obs[0].shape) == 4:
        obs_t = torch.Tensor(np.array(obs)).permute(0,3,1,2)
    else:
        obs_t = torch.Tensor(np.array(obs))

    actions_t = torch.Tensor(np.array(actions)) 
    rewards_t = torch.Tensor(rewards)
    mask = torch.Tensor(mask)

    def Hv(v):
        damping = 1e-2 # Stabilizes error
        kl = policy.kl_divergence(obs_t)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy.parameters(), allow_unused=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping
    
    advantages,returns = estimate_advantage(value_net,obs_t,rewards_t,actions_t,mask,gamma,tau)
    
    values = value_net(obs_t)

    value_optimizer.zero_grad() 
    loss_fn = ((values - returns).pow(2)).mean()
    loss_fn.backward()
    value_optimizer.step()   

    advantages = (advantages - advantages.mean()) / advantages.std() # Normalize advantages


    with torch.no_grad():
        actions_probs_old = policy.log_probs(obs_t,actions_t)
        if isinstance(policy,DiscretePolicy):
            old_policy = policy(obs_t)
        else:
            mean_old, log_std_old, std_old = policy(obs_t)
    
    def loss_fn(grad=True):
        if grad:
            actions_probs = policy.log_probs(obs_t,actions_t)
        else:
            with torch.no_grad():
                actions_probs = policy.log_probs(obs_t,actions_t)
        loss = -advantages * torch.exp(actions_probs - actions_probs_old)
        return loss.mean()

    # Policy gradient with respect to the loss function.
    loss = loss_fn()
    g = torch.autograd.grad(loss, policy.parameters())
    g_vect = torch.cat([grad.contiguous().view(-1) for grad in g])

    # Approximate the inverse Hessian of the KL divergence using CG algorithm.
    direction = conjugate_gradient(Hv, -g_vect,10)
    denom = 0.5*direction.dot(Hv(direction))
    step_size = torch.sqrt(policy.KL_bound/abs(denom))
    full_step = direction * step_size

    old_params = get_model_params(policy)   
    fraction = 1.
    succees = False
    for i in range(10):
        new_params = old_params + fraction * full_step
        set_flat_params_to(policy, new_params)

        if isinstance(policy,DiscretePolicy):
            new_policy = policy(obs_t)
            kl = discrete_kl_divergence(old_policy,new_policy)
        else:
            mean_new, log_std_new, std_new = policy(obs_t)
            kl = gaussian_kl_divergence(mean_old, log_std_old, mean_new, log_std_new)

        kl = kl.mean() 
        loss = loss_fn(False)
        print(f"Loss: {loss.item()}, KL: {kl.item()}")
        if kl < policy.KL_bound:
            print("Reached constraint-solving parameters.")
            succees = True
            break
        fraction = fraction * policy.backtrack_coeff

    if not succees:
        print("Failed to reach constraint-solving parameters.")
        set_flat_params_to(policy, old_params)
    return returns
