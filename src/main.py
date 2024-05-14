from src.data import data_generator, data_normalizer, data_plotter
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import config_parser.model_parser as model_parser
import config_parser.data_parser as data_parser
import utils.train_utils as train_utils


@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project='bayes_by_backprop_test',
               name=f"{cfg.model.type if not cfg.use_ensemble else 'ensemble'}_{cfg.data.name}", config=config_dict)

    if cfg.use_ensemble:
        train_utils.train_ensemble(cfg)
    else:
        train_utils.train_single_model(cfg)

    wandb.finish()
    """
    wandb.init(project='bayes_by_backprop_test',name=cfg.model.type + ' ' + cfg.data.name ,config=config_dict)
    
    # Data generation
    train_data, val_data, test_data = data_parser.get_data(cfg)
    data_plotter.plot_dataset(train_data, 'Trainings data (unnormalized)')

    # Data normalization
    train_data_normalized, val_data_normalized, normalization_params = data_normalizer.normalize_data(train_data, val_data)
    data_plotter.plot_dataset(train_data_normalized, 'Trainings data (normalized)')

    # Model training
    bnn_model = train_model(cfg, train_data_normalized, val_data_normalized)

    # Model evaluation
    mean_values, std_values = evaluate_model(bnn_model, test_data, normalization_params)

    # Plot results
    data_plotter.plot_model_results(test_data[0], test_data[1], train_data, mean_values, std_values,
                                    getattr(data_generator, cfg.data.ground_truth), cfg)

    wandb.finish()
    """

def train_model(cfg, train_data_normalized, val_data_normalized):
    bnn_model = model_parser.get_model(cfg)
    wandb.watch(bnn_model, log='all')
    bnn_model.train_model(train_data_normalized, val_data_normalized, cfg)
    return bnn_model

def evaluate_model(bnn_model, test_data, normalization_params):
    data_plotter.plot_dataset(test_data, 'Test data (unnormalized)')
    x_test_data_normalized, _, _ = data_normalizer.standard_scale(test_data[0],
                                                                  normalization_params['x_train_mean'],
                                                                  normalization_params['x_train_std'])

    mean_values_normalized, std_values_normalized = bnn_model.evaluate_model(x_test_data_normalized)
    mean_values, std_values = data_normalizer.invert_standard_scale(mean_values_normalized, std_values_normalized,
                                                                    normalization_params['y_train_mean'],
                                                                    normalization_params['y_train_std'])
    return mean_values, std_values


if __name__ == '__main__':
    main()