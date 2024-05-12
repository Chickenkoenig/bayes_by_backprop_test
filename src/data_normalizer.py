import torch
import data_generator
import data_plotter

def calculate_statistics(data):
    stats = {
        'mean': torch.mean(data),
        'median': torch.median(data),
        'std': torch.std(data),
        'min_val': torch.min(data),
        'max_val': torch.max(data)
    }
    return stats



def standard_scale(data, mean=None, std=None):
    if mean is None or std is None:
        mean = torch.mean(data)
        std = torch.std(data)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def invert_standard_scale(mean_scaled, std_scaled, original_mean, original_std):
    mean = mean_scaled * original_std + original_mean
    std = std_scaled * original_std
    return mean, std

def main():
    print('hi')



if __name__ == '__main__':
    main()