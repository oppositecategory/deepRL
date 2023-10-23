import torch
from torch.distributions.categorical import Categorical


def policy_gradient_update(network, optimizer, observations, actions, returns):
    obs_tensor = torch.tensor(observations,
                              dtype=torch.float32)
    actions_tensor = torch.tensor(actions,
                                  dtype=torch.int32)
    returns = torch.tensor(returns,
                           dtype=torch.float32)

    optimizer.zero_grad()
    # NOTE: Because we need the backwards() to calculate
    # the gradient of the logits with respect to the network's 
    # parameters, we need to re-evaluate the network at the captured observations as otherwise they are detached.
    logits = network(obs_tensor)
    actions_distribution = Categorical(logits=logits)
    log_probs = actions_distribution.log_prob(actions_tensor)
    batch_loss = -(log_probs * returns).mean()

    batch_loss.backward()
    optimizer.step()