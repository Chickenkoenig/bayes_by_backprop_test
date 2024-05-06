import data_plotter
import data_generator
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import config_parser.model_parser as model_parser

@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(project='bayes_by_backprop_test', config=config_dict)

    data = data_generator.generate_data(getattr(data_generator, cfg.data.noisy_target),
                                        cfg.data.x_start, cfg.data.x_end, cfg.data.num_train_points)
    data_plotter.plot_dataset(data)
    bnn_model = model_parser.get_model(cfg)
    wandb.watch(bnn_model, log='all')
    bnn_model.train_model(data, cfg)

    # generate test data
    x_test, y_test = data_generator.generate_data(getattr(data_generator, cfg.data.noisy_target),
                                        cfg.data.x_start, cfg.data.x_end, cfg.data.num_test_points)

    mean_values, std_values = bnn_model.evaluate_model(x_test)

    data_plotter.plot_model_results(x_test,y_test,mean_values,std_values, getattr(data_generator, cfg.data.ground_truth), cfg)
    wandb.finish()

if __name__ == '__main__':
    main()