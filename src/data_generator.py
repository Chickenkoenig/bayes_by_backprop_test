import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_data(func, x_start, x_end, n_points):
    x = torch.linspace(x_start, x_end, n_points)
    y = func(x)
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    return x, y

def generate_data_with_gaps(func, x_start, x_end, n_points, gap_start, gap_end):
    x = torch.linspace(x_start, x_end, n_points)
    mask = (x < gap_start) | (x > gap_end)
    x = x[mask]
    y = func(x)
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    return x, y

def sin_clean_target(x):
    return torch.sin(np.pi / 2 * x)

def sin_noisy(x):
    y = sin_clean_target(x)
    noise = torch.randn(y.shape) * 0.1
    return y + noise

def sigmoid_mirror_clean_target(x):
    return 2*torch.sigmoid(-7*x)

def sigmoid_mirror_noisy(x):
    y = sigmoid_mirror_clean_target(x)
    noise = torch.randn(y.shape)*0.03
    return y + noise


