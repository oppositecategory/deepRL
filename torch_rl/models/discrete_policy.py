import torch
from torch import nn

class DiscretePolicy(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=64,**kwargs):
        super().__init__()
        self.network = nn.Sequential(
                nn.Linear(input_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,output_dim)
        )
        self.__dict__.update(kwargs)
    
    def forward(self,x):
        x = self.network(x)
        x = torch.softmax(x, dim=1)
        return x
    
    def sample_action(self, x):
        probs = self.forward(x)
        action = probs.multinomial(1)
        return action.item()
    
    def log_probs(self, x,actions):
        probs = self.forward(x)
        return torch.log(probs.gather(1, actions.long().unsqueeze(1)))
    
