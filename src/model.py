import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn


def get_model():
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1, out_features=1000),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1000, out_features=1),
    )
    return model

def train_model(model, data, epochs, learning_rate=0.01, kl_weight=0.01):
    x, y = data
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    for step in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        mse = mse_loss(predictions, y)
        kl = kl_loss(model)
        cost = mse + kl_weight * kl
        cost.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f'Epoch{step}: MSE={mse.item():.2f}, KL={kl.item():.2f}')