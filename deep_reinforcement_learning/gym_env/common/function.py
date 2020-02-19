import torch.nn as nn

def mlp(layer_sizes, activation, out_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes)-1):
        tmp_act = activation if i < len(layer_sizes)-2 else out_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), tmp_act()]
    return nn.Sequential(*layers)
