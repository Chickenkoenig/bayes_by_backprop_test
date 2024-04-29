import torch
import matplotlib.pyplot as plt

def generate_data(func, x_start, x_end, n_points):
    x = torch.linspace(x_start, x_end, n_points)
    y = func(x)
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    return x, y


def clean_target(x):
    return x.pow(5) -10* x.pow(1)+1

def noisy_target(x):
    return x.pow(5) -10* x.pow(1) + 2*torch.rand(x.size())


def main():
    x, y = generate_data(noisy_target, -2, 2, 500)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()

if __name__ == "__main__":
    main()