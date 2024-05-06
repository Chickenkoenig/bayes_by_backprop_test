import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb

class MCDropoutLayer(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(MCDropoutLayer, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.training:
            return F.dropout(x, self.p, True, self.inplace)
        return F.dropout(x, self.p, self.training, self.inplace)

class MCDropoutNet(nn.Module):
    def __init__(self, cfg):
        super(MCDropoutNet, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in cfg.model.layers:
            if layer_cfg.type == 'Linear':
                layer = nn.Linear(
                    in_features=layer_cfg.in_features,
                    out_features=layer_cfg.out_features
                )
            elif layer_cfg.type == 'ReLU':
                layer = nn.ReLU()
            elif layer_cfg.type == 'MCDropoutLayer':
                layer = MCDropoutLayer(cfg.model.dropout_prob)
            else:
                raise ValueError(f"Unsupported layer type: {layer_cfg.type}")
            self.layers.append(layer)
        self.model = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.model(x)

    def train_model(self, data, cfg):
        x, y = data
        optimizer = optim.Adam(self.parameters(), lr=cfg.train.learning_rate)
        mse_loss = nn.MSELoss()

        for step in range(cfg.train.epochs):
            self.train()
            optimizer.zero_grad()
            predictions = self(x)
            cost = mse_loss(predictions, y)
            cost.backward()
            optimizer.step()

            wandb.log({
                "epoch": step,
                "mse_cost": cost.item()
            })
            print(f'Epoch{step}: MSE={cost.item():.2f}')


    def evaluate_model(self, x_test, num_samples=10000):
        # performs 10000 forward passes for each test data point through the trained model
        models_result = np.array([self(x_test).data.numpy() for k in range(num_samples)])
        models_result = models_result[:, :, 0]
        models_result = models_result.T

        # Computes the mean and standard deviation across the 10000 for each test input
        mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
        std_values = np.array([models_result[i].std() for i in range(len(models_result))])
        return mean_values, std_values


