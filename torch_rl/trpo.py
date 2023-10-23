import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian

from .utils import *

class ConvPolicy(ConvNet):
    def __init__(self, 
                input_dim,
                output_dim,
                KL_bound=0.001,
                backtrack_coeff = 0.8,
                filters=[(16, 4,2)],   
                hidden_dim=128):
        super().__init__(input_dim,output_dim,filters,hidden_dim)
        self.KL_bound = torch.Tensor([KL_bound])
        self.backtrack_coeff = torch.Tensor([backtrack_coeff])

    def sample_action(self, x):
        logits = self.forward(x)
        action = Categorical(logits=logits).sample().item()
        return action
    
    def log_probs(self, x,actions_t):
        logits = self.forward(x)
        print("Logits:",logits)
        actions_distribution = Categorical(logits=logits)
        log_probs = actions_distribution.log_prob(actions_t)
        return log_probs

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=64,
                 KL_bound=0.001,
                 backtrack_coeff = 0.8):
        super().__init__()
        self.KL_bound = torch.Tensor([KL_bound])
        self.backtrack_coeff = torch.Tensor([backtrack_coeff])

        self.layer1 =  nn.Linear(input_dim,hidden_dim)
        self.layer2 =  nn.Linear(hidden_dim,hidden_dim)
        self.layer3 =  nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = torch.softmax(x, dim=1)
        return x
    
    def sample_action(self, x):
        probs = self.forward(x)
        action = probs.multinomial(1)
        return action.item()
    
    def log_probs(self, x,actions_t):
        probs = self.forward(x)
        return torch.log(probs.gather(1, actions_t.long().unsqueeze(1)))

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

    def kl_fn():
        """
        NOTE: The KL Divergence is only needed for it's Hessian and it's evaluated at the old policy parameters.
        """
        action_probs1 = policy(obs_t)
        action_probs0 = action_probs1.data
        kl = action_probs0 * (torch.log(action_probs0) - torch.log(action_probs1))
        return kl.sum(1,keepdim=True)


    def Hv(v):
        """
        This function implements a neat trick to calculate H@v where H=the Hessian of the averaged KL divergence.
        Considering we are only interested in solution to Hs=g using conjugate gradient, we are basically only
        intersted in the matrix-product Hx and not H itself. Hence it can be shown that the Hessian-vector product 
        is equal to derivative of the product of the first derivative of KL w.r.t to parameters multiplied by the input.
        """
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
        old_policy = policy(obs_t)
        actions_probs_old = policy.log_probs(obs_t,actions_t)
    

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

    # To handle cases of NaN resulting from floating-point errors it seems
    # it's better to swap the denomanator and numerator to reduce risk of NaN.
    # The numerator is 0.02~delta*2 which is >>> then the gradients with higher probability for exploding.
    denom = 0.5*torch.dot(direction,-g_vect)

    step_size = torch.sqrt(policy.KL_bound/denom)      
    full_step = direction * step_size

    old_params = get_model_params(policy)   
    fraction = 1.
    succees = False
    for i in range(10):
        new_params = old_params + fraction * full_step
        set_flat_params_to(policy, new_params)
        new_policy = policy(obs_t)
        loss = loss_fn(False)
        kl = old_policy * torch.log(old_policy/new_policy).sum(1,keepdim=True)
        kl = kl.mean()
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
