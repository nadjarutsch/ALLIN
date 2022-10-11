import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        nodes_in = [n_input] + n_hidden
        nodes_out = n_hidden + [n_output]
        
        self.layers = nn.ModuleList()
        for inputs, outputs in zip(nodes_in[:-1], nodes_out[:-1]):
            self.layers.append(nn.Linear(inputs, outputs)) 
            self.layers.append(nn.ReLU())     
        
        self.layers.append(nn.Linear(nodes_in[-1], nodes_out[-1]))
    
    def forward(self, x):
        for layer in self.layers:     
            x = layer(x)
        return x.squeeze()
