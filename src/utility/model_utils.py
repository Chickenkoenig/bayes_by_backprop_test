from src.config_parser import model_parser
from src.data import data_plotter, data_normalizer
import wandb
import torch


def train_model(cfg, train_data_normalized, val_data_normalized):
    bnn_model = model_parser.get_model(cfg)
    wandb.watch(bnn_model, log='all')
    bnn_model.train_model(train_data_normalized, val_data_normalized, cfg)
    return bnn_model

def evaluate_mlp_ensemble(models, test_data, normalization_params):
    x_test_data_normalized, _, _ = data_normalizer.standard_scale(test_data[0],
                                                                  normalization_params['x_train_mean'],
                                                                  normalization_params['x_train_std'])
    with torch.no_grad():
        # Generate predictions from each model in the ensemble
        predictions = [model(x_test_data_normalized) for model in models]
        # Stack all predictions for aggregation
        stacked_predictions = torch.stack(predictions, dim=0)
        mean_values_normalized = stacked_predictions.mean(dim=0).squeeze()
        std_values_normalized = stacked_predictions.std(dim=0).squeeze()
        mean_values, std_values = data_normalizer.invert_standard_scale(mean_values_normalized, std_values_normalized,
                                                                        normalization_params['y_train_mean'],
                                                                        normalization_params['y_train_std'])
        return mean_values, std_values

def evaluate_bayesian_model(bnn_model, test_data, normalization_params):

    x_test_data_normalized, _, _ = data_normalizer.standard_scale(test_data[0],
                                                                  normalization_params['x_train_mean'],
                                                                  normalization_params['x_train_std'])

    mean_values_normalized, std_values_normalized = bnn_model.evaluate_model(x_test_data_normalized)
    mean_values, std_values = data_normalizer.invert_standard_scale(mean_values_normalized, std_values_normalized,
                                                                    normalization_params['y_train_mean'],
                                                                    normalization_params['y_train_std'])
    return mean_values, std_values