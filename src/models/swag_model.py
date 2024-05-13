import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class SwagModel(nn.Module):
    def __init__(self, cfg):
        super(SwagModel, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in cfg.model.layers:
            if layer_cfg.type == 'Linear':
                layer = nn.Linear(
                    in_features=layer_cfg.in_features,
                    out_features=layer_cfg.out_features
                )
            elif layer_cfg.type == 'ReLU':
                layer = nn.ReLU()
            else:
                raise ValueError(f"Unsupported layer type: {layer_cfg.type}")
            self.layers.append(layer)
        self.model = nn.Sequential(*self.layers)

        self.swag_start = cfg.model.swag_start
        self.update_interval = cfg.model.update_interval
        self.steps = 0
        self.mean = None
        self.sq_means = None
        self.deviations = []

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_data, val_data, cfg):
        x, y = train_data
        x_val, y_val = val_data
        optimizer = optim.SGD(self.parameters(), lr=cfg.train.learning_rate)
        mse_loss = nn.MSELoss()

        for epoch in range(cfg.train.epochs):
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
                "epoch": epoch,
                "train_loss": loss_train.item(),
                "val_loss": loss_val.item()
            })
            print(f'Epoch{epoch}: Train MSE={loss_train.item():.2f}, Val MSE={loss_val.item():.2f}')
            self.swag_update()

    def swag_update(self):
        self.steps += 1
        if self.steps >= self.swag_start:
            params = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
            if self.mean is None:
                self.mean = params.clone()
                self.sq_means = params.square()
            else:
                self.mean = (self.mean * (self.steps - self.swag_start) + params) / (self.steps - self.swag_start + 1)
                self.sq_means = (self.sq_means * (self.steps - self.swag_start) + params.square()) / (self.steps - self.swag_start + 1)
            self.deviations.append(params - self.mean)



    def sample_parameters(self):
        mean = self.mean
        variance = torch.relu(self.sq_means - self.mean.square())
        std = variance.sqrt()
        sampled_params = torch.normal(mean, std)
        torch.nn.utils.vector_to_parameters(sampled_params, self.model.parameters())


    def evaluate_model(self, x_test, num_samples=10000):
        predictions = []
        for _ in range(num_samples):
            self.sample_parameters()
            with torch.no_grad():
                preds = self.model(x_test)
            predictions.append(preds)
        predictions = torch.stack(predictions)
        mean_values = predictions.mean(dim=0).squeeze()
        std_values = predictions.std(dim=0).squeeze()
        return mean_values, std_values