import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import numpy as np
import wandb


class BBBModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in cfg.model.layers:
            if layer_cfg.type == 'BayesLinear':
                layer = bnn.BayesLinear(
                    prior_mu=layer_cfg.prior_mu,
                    prior_sigma=layer_cfg.prior_sigma,
                    in_features=layer_cfg.in_features,
                    out_features=layer_cfg.out_features
                )
            elif layer_cfg.type == 'ReLU':
                layer = nn.ReLU()
            else:
                raise ValueError(f"Unsupported layer type: {layer_cfg.type}")
            self.layers.append(layer)
        self.model = nn.Sequential(*self.layers)
        self.kl_weight = cfg.model.kl_weight

    def forward(self, x):
        return self.model(x)

    def train_model(self, data, epochs, cfg):
        x, y = data
        optimizer = optim.Adam(self.parameters(), lr=cfg.train.learning_rate)
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        for step in range(cfg.train.epochs):
            self.train()
            optimizer.zero_grad()
            predictions = self(x)
            mse = mse_loss(predictions, y)  # represents the likelihood term logP(D|w) in the BBB approximation
            kl = kl_loss(self.model)  # measures divergence between the posterior distr of the weights and the prior
            cost = mse + self.kl_weight * kl
            cost.backward()
            optimizer.step()

            if step % 100 == 0 or step == epochs - 1:
                wandb.log({
                    "epoch": step,
                    "mse": mse.item(),
                    "kl_divergence": kl.item(),
                    "total_cost": cost.item()
                })
                print(f'Epoch{step}: MSE={mse.item():.2f}, KL={kl.item():.2f}')

    def evaluate_model(self, x_test, num_samples=10000):
        # performs 10000 forward passes for each test data point through the trained model
        models_result = np.array([self(x_test).data.numpy() for k in range(num_samples)])
        models_result = models_result[:, :, 0]
        models_result = models_result.T

        # Computes the mean and standard deviation across the 10000 for each test input
        mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
        std_values = np.array([models_result[i].std() for i in range(len(models_result))])
        return mean_values, std_values