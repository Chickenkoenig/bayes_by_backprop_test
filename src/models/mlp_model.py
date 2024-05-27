import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class MLPModel(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = layers
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)