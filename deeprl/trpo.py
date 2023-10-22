import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd.functional import hessian

from .utils import *

class Policy(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 KL_bound=0.001,
                 backtrack_coeff = 0.8,
                 filters=[(16, 4,2)],   
                 hidden_dim=256):
        super().__init__()
        self.KL_bound = torch.Tensor([KL_bound])
        self.backtrack_coeff = torch.Tensor([backtrack_coeff])

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
        # x = x.view(-1,self.conv_rep)
        x = torch.flatten(x,1,-1)
        x = self.fc_layer(x)
        return x
    
    def sample_action(self, x):
        logits = self.forward(x)
        action = Categorical(logits=logits).sample().item()
        return action
    
    def log_probs(self, x,actions_t):
        logits = self.forward(x)
        actions_distribution = Categorical(logits=logits)
        log_probs = actions_distribution.log_prob(actions_t)
        return log_probs

@torch.no_grad()
def estimate_advantage(value_net,states, rewards, actions, masks,gamma):
    # Estimating the advantage using a Value func approximator and TD difference.
    values = value_net(torch.flatten(states,1,-1))
    advantages = torch.zeros(states.size(0),1)

    for i in range(states.size(0)-1):
        advantages[i,0] = rewards[i] + gamma* values[i+1] - masks[i] * values[i]
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
        alpha = torch.dot(r,r) / torch.dot(p, _Hv)
        x += alpha*p 
        r_next = r - alpha*_Hv
        if torch.norm(r) < threshold:
            break 
        beta = torch.dot(r_next,r_next) / torch.dot(r,r)
        p = r_next + beta*p 
        r = r_next
    return x 

@torch.no_grad()
def backtrack_line_search(model,f,grad_f,x,p,alpha_0,c=0.1,max=10):
    """ Implements backtrack line search.
        Args:
            - f: the function we wish to optimize
            - grad_f: the gradient of f 
            - x: starting position
            - p: search direction
            - alpha_0: initial step size guess
            - c: control parameter

        This is a bit of a specific variation of the algorithm. Instead of introducting another parameter tau as the shrinkage pace we use exponents of alpha and return the smallest exponent j, such as alpha^j minmize the function as desired.

        TODO: Detect bug in unchanged loss. It might be because the internal parameters aren't actually reached by set_flat_params.
    """
    m = torch.dot(grad_f,p)
    alpha = alpha_0
    initial_loss = f(False).data
    t = -c*m
    for j in range(max):
        candidate_params = x + alpha*p
        set_flat_params_to(model, candidate_params)
        curr_loss = f(False)
        print(f"{j}/{max} Current loss: {curr_loss}")
        if initial_loss - curr_loss >= alpha*t:
            return True, candidate_params
        alpha = alpha*alpha
    return False,x + p


def trpo_update(policy, value_net, observations, actions, returns, mask,gamma):
    actions_t = torch.Tensor(np.array(actions)) 
    rewards_t = torch.Tensor(returns)
    obs_t = torch.Tensor(np.array(observations)).permute(0,3,1,2)
    mask = torch.Tensor(mask)

    def kl_fn():
        """
        NOTE: The KL Divergence is only needed for it's Hessian and it's evaluated at the old policy parameters.
              Observe that action_probs1.data is detaching the gradient from the tensor and hence this function is
              merely used for evaluating the Hessian at the current policy.
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
        grads = torch.autograd.grad(kl_v, policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping
    
    advantages = estimate_advantage(value_net,obs_t,rewards_t,actions_t,mask,gamma)
    advantages = (advantages - advantages.mean()) / advantages.std() # Normalize advantages
    actions_probs = policy.log_probs(obs_t,actions_t)
    actions_probs_old = actions_probs.data

    def loss_fn(grad=True):
        if grad:
            actions_probs = policy.log_probs(obs_t,actions_t)
        else:
            with torch.no_grad():
                actions_probs = policy.log_probs(obs_t,actions_t)
        return -(torch.exp(actions_probs - actions_probs_old) * advantages).mean()

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
    step_size = torch.sqrt(denom/policy.KL_bound)      
    
    if torch.isnan(step_size):
        print("Floating point error using LR=0.001")
        full_step = 1e-3 * direction
    else:
        full_step = direction / step_size

    old_params = get_model_params(policy)   
    success, new_params= backtrack_line_search(policy, 
                                               loss_fn,
                                               -g_vect,
                                               old_params,
                                               full_step,
                                               policy.backtrack_coeff)

    set_flat_params_to(policy, new_params)
    print(f"grad_norm: {-g_vect.norm()} average return: {rewards_t.mean()}")
    return loss