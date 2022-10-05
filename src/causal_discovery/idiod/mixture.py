import torch.nn as nn


class IdentityMixture(nn.Module):
    def __init__(self, single_target: bool = True):
        super().__init__()
        self.single_target = single_target

    def forward(self, mixture_input):
        probs = 1 - mixture_input[..., 1:] if self.single_target else mixture_input
        return probs


class MLPMixture(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, single_target=False, bias=True):
        super().__init__()
        self.single_target = single_target
        nodes_in = [n_input] + n_hidden
        nodes_out = n_hidden + [n_output]

        self.layers = nn.ModuleList()
        for inputs, outputs in zip(nodes_in[:-1], nodes_out[:-1]):
            self.layers.append(nn.Linear(inputs, outputs, bias=bias))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(nodes_in[-1], nodes_out[-1]))
        if self.single_target:
            self.layers.append(nn.Softmax())
        else:
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        probs = 1 - x[..., 1:] if self.single_target else x
        return probs.squeeze()


