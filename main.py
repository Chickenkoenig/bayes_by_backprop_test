import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt

def create_dataset(n_points):
    x = torch.linspace(-2, 2, n_points)
    y = x.pow(5) -10* x.pow(1) + 2*torch.rand(x.size())
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    return x, y

def create_testset(n_points):
    x_test = torch.linspace(-2, 2, n_points)
    y_test = target(x_test)
    x_test = torch.unsqueeze(x_test, dim=1)
    y_test = torch.unsqueeze(y_test, dim=1)
    return x_test, y_test

def plot_dataset(x, y):
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.title("Scatter plot of dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# ground truth generator
def clean_target(x):
    return x.pow(5) -10* x.pow(1)+1

#noisy data generator
def target(x):
    return x.pow(5) -10* x.pow(1) + 2*torch.rand(x.size())

def get_model():
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1, out_features=1000),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1000, out_features=1),
    )
    return model

def main():
    x, y = create_dataset(500)
    plot_dataset(x, y)
    model = get_model()
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for step in range(2000):
        pre = model(x)
        mse = mse_loss(pre, y)
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

    x_test, y_test = create_testset(300)


    models_result = np.array([model(x_test).data.numpy() for k in range(10000)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T
    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])

    #plot confidence intervalls
    plt.figure(figsize=(10, 8))
    plt.plot(x_test.data.numpy(), mean_values, color='navy', lw=3, label='Predicted Mean Model')
    plt.fill_between(x_test.data.numpy().T[0], mean_values - 3.0 * std_values, mean_values + 3.0 * std_values,
                     alpha=0.2, color='navy', label='99.7% confidence interval')
    # plt.plot(x_test.data.numpy(),mean_values,color='darkorange')
    plt.plot(x_test.data.numpy(), y_test.data.numpy(), '.', color='darkorange', markersize=4, label='Test set')
    plt.plot(x_test.data.numpy(), clean_target(x_test).data.numpy(), color='green', markersize=4,
             label='Target function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main()