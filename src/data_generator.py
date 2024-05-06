import torch
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

def linear_func_clean_target(x):
    return 3 * x + 5

def linear_func_noisy(x):
    y = linear_func_clean_target(x)
    noise = torch.randn(y.shape) * 1.25  # base noise
    high_noise_indices = (x > 4) & (x < 6)  # increasing noise between x = 4 and x = 6
    noise[high_noise_indices] *= 5  # triple the noise in the noisy region
    return  y + noise


def clean_target(x):
    return x.pow(5) -10* x.pow(1)


def noisy_target(x):
    base_noise = 2 * (torch.rand(x.size()) - 0.5)
    return x.pow(5) -10* x.pow(1) + base_noise



def noisy_target_high_noise_area(x):
    y = clean_target(x)
    noise = torch.randn(y.shape) * 1.25  # base noise
    high_noise_indices = (x > -0.5) & (x < 0.5)  # increasing noise between x = -0.5 and x = 0.5
    noise[high_noise_indices] *= 5
    return y + noise


