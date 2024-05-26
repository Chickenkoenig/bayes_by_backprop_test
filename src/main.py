import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import utils.train_utils as train_utils


@hydra.main(config_path="../config2", config_name="main", version_base="1.1")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project='test_uncertainty_estimation',
               name=f"{cfg.model.type if not cfg.use_ensemble else 'ensemble'}_{cfg.data.name}", config=config_dict)

    if cfg.use_ensemble:
        train_utils.train_ensemble(cfg)
    else:
        train_utils.train_single_model(cfg)

    wandb.finish()


if __name__ == '__main__':
    main()
