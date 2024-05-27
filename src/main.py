import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import utils.train_utils as train_utils
import os
from config_parser.parser import parse_config

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    #wandb.init(project='test_uncertainty_estimation',
     #          name=cfg.exp_name, config=config_dict)
    """
    if cfg.use_ensemble:
        train_utils.train_ensemble(cfg)
    else:
        train_utils.train_single_model(cfg)
    """
    data, models = parse_config(cfg)
    print(models)
    #wandb.finish()


if __name__ == '__main__':

    main()
