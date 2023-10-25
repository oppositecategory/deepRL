import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian

from .models.continuous_policy import ContinuousPolicy
from .models.discrete_policy import DiscretePolicy
from .utils import *

@torch.no_grad()
def estimate_advantage(value_net,states, rewards, actions, masks,gamma,tau):
    values = value_net(states)

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

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

    def kl_d():
        action_probs1 = policy(obs_t)
        action_probs0 = action_probs1.data
        kl = action_probs0 * (torch.log(action_probs0) - torch.log(action_probs1))
        return kl.sum(1,keepdim=True)
    
    def kl_c():
        mean, log_std, std = policy(obs_t)
        mean0, log_std0 , std0 = mean.data, log_std.data, std.data
        var, var0 = std.pow(2), std0.pow(2)
        kl = log_std - log_std0 + (mean0-mean).pow(2)/ (2*std.pow(2))
        return kl.sum(1,keepdim=True)
    
    if isinstance(policy, DiscretePolicy):
        kl_fn = kl_d
    else:
        kl_fn = kl_c

    def Hv(v):
        damping = 1e-2 # Stabilizes error
        kl = kl_fn()
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
        # Note that in case of continous policy the old_policy variable will be a tuple
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
    denom = 0.5*torch.dot(direction,-g_vect)
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
            kl = old_policy * torch.log(old_policy/new_policy).sum(1,keepdim=True)
        else:
            mean_new, log_std_new, std_new = policy(obs_t)
            var_old, var_new = std_old.pow(2), std_new.pow(2)
            new_probs = policy.log_probs(obs_t,actions_t)
            # kl = log_std_old - log_std_new + (std_old.pow(2) + (mean_old - mean_new).pow(2)) / (2.0 * std_new.pow(2)) - 0.5
            kl = log_std_new - log_std_old + (mean_new-mean_old).pow(2)/ ( 2*log_std_old.pow(2))
            kl = kl.sum(1,keepdim=True)

        kl = kl.mean(dim=0) 
        loss = loss_fn(False)
        print(f"Loss: {loss.item()}, KL: {kl.item()}")
        if kl < policy.KL_bound and kl > 0:
            print("Reached constraint-solving parameters.")
            succees = True
            break
        elif kl < 0:
            break
        fraction = fraction * policy.backtrack_coeff

    if not succees:
        print("Failed to reach constraint-solving parameters.")
        set_flat_params_to(policy, old_params)
    return returns
