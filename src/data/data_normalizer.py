import torch


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

def normalize_data(train_data, val_data):
    x_train_data_normalized, x_train_mean, x_train_std = standard_scale(train_data[0])
    y_train_data_normalized, y_train_mean, y_train_std = standard_scale(train_data[1])
    x_val_data_normalized, _, _ = standard_scale(val_data[0], x_train_mean, x_train_std)
    y_val_data_normalized, _, _ = standard_scale(val_data[1], y_train_mean, y_train_std)

    train_data_normalized = x_train_data_normalized, y_train_data_normalized
    val_data_normalized = x_val_data_normalized, y_val_data_normalized

    normalization_params = {
        'x_train_mean': x_train_mean,
        'x_train_std': x_train_std,
        'y_train_mean': y_train_mean,
        'y_train_std': y_train_std
    }
    return train_data_normalized, val_data_normalized, normalization_params

def main():
    print('hi')



if __name__ == '__main__':
    main()