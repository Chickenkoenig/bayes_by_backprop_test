import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import matplotlib.pyplot as plt

class MCDropoutLayer(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(MCDropoutLayer, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)


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


    def train_model(self, train_data, val_data, cfg):
        x, y = train_data
        x_val, y_val = val_data
        optimizer = optim.Adam(self.parameters(), lr=cfg.train.learning_rate, weight_decay=1e-6)
        mse_loss = nn.MSELoss()

        for step in range(cfg.train.epochs):
            self.train()
            optimizer.zero_grad()
            predictions = self(x)
            loss_train = mse_loss(predictions, y)
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                predictions_val = self(x_val)
                loss_val = mse_loss(predictions_val, y_val)

            wandb.log({
                "epoch_dropout": step,
                "train_loss_dropout": loss_train.item(),
                "val_loss_dropout": loss_val.item()
            })
            print(f'Epoch{step}: Train MSE={loss_train.item():.2f}, Val MSE={loss_val.item():.2f}')


    def evaluate_model(self, x_test, num_samples=5000):
        """"
        # performs 10000 forward passes for each test data point through the trained model
        models_result = np.array([self(x_test).data.numpy() for k in range(num_samples)])
        models_result = models_result[:, :, 0]
        models_result = models_result.T

        # Computes the mean and standard deviation across the 10000 for each test input
        mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
        std_values = np.array([models_result[i].std() for i in range(len(models_result))])
        return mean_values, std_values
        """
        self.train()  # Ensure dropout is active
        with torch.no_grad():
            predictions = torch.stack([self(x_test) for _ in range(num_samples)], dim=0)
        mean_values = predictions.mean(dim=0).squeeze()
        std_values = predictions.std(dim=0).squeeze()

        """
        # Plotting
        plt.figure(figsize=(12, 8))
        x_test_np = x_test.numpy()  # Ensure x_test is a numpy array for plotting
        predictions_np = predictions.numpy()  # Convert predictions to numpy array

        # Create a scatter plot for each input across all samples
        for i in range(x_test.shape[0]):
            plt.scatter([x_test_np[i]] * num_samples, predictions_np[:, i], alpha=0.1, marker='.', color='blue')

        plt.xlabel('Input value')
        plt.ylabel('Predicted outputs')
        plt.title('Scatter Plot of All Predictions for Each Input')
        plt.show()
        """
        return mean_values, std_values


