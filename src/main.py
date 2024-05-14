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



if __name__ == '__main__':
    main()