import data_plotter
import data_generator
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import config_parser.model_parser as model_parser
import config_parser.data_parser as data_parser
import data_normalizer




@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project='bayes_by_backprop_test',name=cfg.model.type + ' ' + cfg.data.name ,config=config_dict)
    # data generation
    train_data, val_data, test_data = data_parser.get_data(cfg)
    data_plotter.plot_dataset(train_data, 'Trainings data (unnormalized)')

    # normalize train data
    x_train_data_normalized, x_train_mean, x_train_std = data_normalizer.standard_scale(train_data[0])
    y_train_data_normalized, y_train_mean, y_train_std = data_normalizer.standard_scale(train_data[1])
    x_val_data_normalized, _, _ = data_normalizer.standard_scale(val_data[0], x_train_mean, x_train_std)
    y_val_data_normalized, _, _ = data_normalizer.standard_scale(val_data[1], y_train_mean, y_train_std)

    train_data_normalized = x_train_data_normalized, y_train_data_normalized
    data_plotter.plot_dataset(train_data_normalized, 'Trainings data (normalized)')
    val_data_normalized = x_val_data_normalized, y_val_data_normalized

    # model and training
    bnn_model = model_parser.get_model(cfg)
    wandb.watch(bnn_model, log='all')
    bnn_model.train_model(train_data_normalized, val_data_normalized, cfg)

    # normalize test data
    data_plotter.plot_dataset(test_data, 'Test data (unnormalized)')
    x_test_data_normalized, _, _ = data_normalizer.standard_scale(test_data[0], x_train_mean, x_train_std)

    # evaluate model on test data
    mean_values_normalized, std_values_normalized = bnn_model.evaluate_model(x_test_data_normalized)

    # denormalize predicted means and standard deviations
    mean_values, std_values = data_normalizer.invert_standard_scale(mean_values_normalized, std_values_normalized,
                                                                    y_train_mean, y_train_std)

    # plot results
    data_plotter.plot_model_results(test_data[0], test_data[1], train_data, mean_values, std_values,
                                    getattr(data_generator, cfg.data.ground_truth), cfg)


    wandb.finish()



if __name__ == '__main__':
    main()