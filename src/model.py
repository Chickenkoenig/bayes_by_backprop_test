import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import numpy as np
import wandb

def get_model(cfg):
    layers = []
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
        layers.append(layer)

    model = nn.Sequential(*layers)
    return model

def train_model(model, data, epochs, learning_rate=0.01, kl_weight=0.01):
    x, y = data
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    for step in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x) #are here multiple predictions computed or does it have to be written manually?
        mse = mse_loss(predictions, y) #represents the likelihood term logP(D|w) in the BBB approximation
        kl = kl_loss(model) #measures divergence between the posterior distr of the weights and the prior
        cost = mse + kl_weight * kl
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

def evaluate_model(model,x_test, num_samples=10000):
    # performs 10000 forward passes for each test data point through the trained model
    models_result = np.array([model(x_test).data.numpy() for k in range(num_samples)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    # Computes the mean and standard deviation across the 10000 for each test input
    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])
    return mean_values, std_values