import data_plotter
import data_generator
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import config_parser.model_parser as model_parser
import config_parser.data_parser as data_parser




@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(project='bayes_by_backprop_test',name=cfg.model.type + ' ' + cfg.data.name ,config=config_dict)
    train_data, validation_data, test_data = data_parser.get_data(cfg)
    data_plotter.plot_dataset(train_data)
    bnn_model = model_parser.get_model(cfg)
    wandb.watch(bnn_model, log='all')
    bnn_model.train_model(train_data, validation_data, cfg)

    data_plotter.plot_dataset(test_data)
    x_test, y_test = test_data
    mean_values, std_values = bnn_model.evaluate_model(x_test)



    data_plotter.plot_model_results(x_test, y_test, train_data, mean_values, std_values, getattr(data_generator, cfg.data.ground_truth), cfg)
    wandb.finish()

if __name__ == '__main__':
    main()