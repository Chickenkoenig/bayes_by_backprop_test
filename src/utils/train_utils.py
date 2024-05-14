from omegaconf import OmegaConf
import torch
from src.utils import data_utils, model_utils, plot_utils
from hydra.utils import to_absolute_path


def train_single_model(cfg):
    train_data, val_data, test_data = data_utils.prepare_data(cfg)
    plot_utils.plot_data(train_data, 'Trainings data (unnormalized)')
    train_data_normalized, val_data_normalized, normalization_params = data_utils.normalize_data(train_data, val_data)
    plot_utils.plot_data(train_data_normalized, 'Trainings data (normalized)')
    plot_utils.plot_data(test_data, 'Test data (unnormalized)')
    bnn_model = model_utils.train_model(cfg, train_data_normalized, val_data_normalized)
    mean_values, std_values = model_utils.evaluate_model(bnn_model, test_data, normalization_params)
    plot_utils.plot_results(test_data, train_data, mean_values, std_values, cfg)

def train_ensemble(cfg):
    train_data, val_data, test_data = data_utils.prepare_data(cfg)
    plot_utils.plot_data(train_data, 'Trainings data (unnormalized)')
    train_data_normalized, val_data_normalized, normalization_params = data_utils.normalize_data(train_data, val_data)
    plot_utils.plot_data(train_data_normalized, 'Trainings data (normalized)')
    plot_utils.plot_data(test_data, 'Test data (unnormalized)')


    ensemble_predictions = []
    for model_name in cfg.ensemble.models:
        # Load model-specific configuration
        model_cfg_path = to_absolute_path(f"config/model/{model_name}.yaml")
        model_cfg = OmegaConf.load(model_cfg_path)

        # Load training-specific configuration
        train_cfg_path = to_absolute_path(f"config/train/{model_name}/{cfg.data.name}.yaml")
        train_cfg = OmegaConf.load(train_cfg_path)

        # Merge base configuration with model and train configurations
        OmegaConf.set_struct(cfg, False)
        merged_cfg = OmegaConf.merge(cfg, {"model": model_cfg}, {"train": train_cfg})
        OmegaConf.set_struct(cfg, True)
        print(OmegaConf.to_yaml(merged_cfg))

        bnn_model = model_utils.train_model(merged_cfg, train_data_normalized, val_data_normalized)
        mean_values, std_values = model_utils.evaluate_model(bnn_model, test_data, normalization_params)
        ensemble_predictions.append((mean_values, std_values))

    aggregate_and_plot_results(test_data, train_data, ensemble_predictions, cfg)

def aggregate_and_plot_results(test_data, train_data, ensemble_predictions, cfg):
    mean_tensors = torch.stack([pred[0] for pred in ensemble_predictions])
    std_tensors = torch.stack([pred[1] for pred in ensemble_predictions])

    ensemble_mean = mean_tensors.mean(dim=0)
    ensemble_var = std_tensors.pow(2).mean(dim=0) + mean_tensors.var(dim=0, unbiased=False)
    ensemble_std = ensemble_var.sqrt()

    plot_utils.plot_results(test_data, train_data, ensemble_mean, ensemble_std, cfg)
