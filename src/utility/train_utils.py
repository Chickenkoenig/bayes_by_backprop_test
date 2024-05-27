from omegaconf import OmegaConf
import torch
from . import data_utils, model_utils, plot_utils
from hydra.utils import to_absolute_path
import wandb

def train_models(cfg, data, models):
    train_data, val_data, test_data = data
    plot_utils.plot_data(train_data, 'Trainings data (unnormalized)')
    train_data_normalized, val_data_normalized, normalization_params = data_utils.normalize_data(train_data, val_data)
    plot_utils.plot_data(train_data_normalized, 'Trainings data (normalized)')

    # training
    for model in models:
        wandb.watch(model, log='all')
        model.train_model(train_data_normalized, val_data_normalized, cfg)

    evaluate_models(cfg, test_data, train_data, models, normalization_params)

def evaluate_models(cfg, test_data, train_data, models, normalization_params):
    # evaluation
    predictions = []

    if cfg.model.architecture.type == 'mlp':

        mean_values, std_values = model_utils.evaluate_mlp_ensemble(models, test_data, normalization_params)
        plot_utils.plot_results(test_data, train_data, mean_values, std_values, cfg)
    else:

        for model in models:
            mean_values, std_values = model_utils.evaluate_bayesian_model(model, test_data, normalization_params)
            predictions.append((mean_values, std_values))

        aggregate_and_plot_results(test_data, train_data, predictions, cfg)

#function for aggregating bayesian ensemble predictions
def aggregate_and_plot_results(test_data, train_data, ensemble_predictions, cfg):
    mean_tensors = torch.stack([pred[0] for pred in ensemble_predictions])
    std_tensors = torch.stack([pred[1] for pred in ensemble_predictions])

    ensemble_mean = mean_tensors.mean(dim=0)
    ensemble_var = std_tensors.pow(2).mean(dim=0) + mean_tensors.var(dim=0, unbiased=False)
    ensemble_std = ensemble_var.sqrt()

    plot_utils.plot_results(test_data, train_data, ensemble_mean, ensemble_std, cfg)
