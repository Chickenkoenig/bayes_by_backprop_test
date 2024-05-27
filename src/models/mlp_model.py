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

    def train_model(self, data, val_data, cfg):
        x, y = data
        x_val, y_val = val_data
        optimizer = optim.Adam(self.parameters(), lr=cfg.train.learning_rate)
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
                "epoch_mlp": step,
                "train_loss_mlp": loss_train.item(),
                "val_loss_mlp": loss_val.item()
            })
            print(f'Epoch{step}: Train MSE={loss_train.item():.2f}, Val MSE={loss_val.item():.2f}')

