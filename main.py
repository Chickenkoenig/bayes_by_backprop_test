import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import src.utility.train_utils as train_utils
from src.config_parser.parser import parse_config




@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(project=cfg.recorder.wandb.project,
               job_type=cfg.recorder.wandb.job_type,
               group=cfg.recorder.wandb.group,
               name=cfg.recorder.wandb.run_name,
               config=config_dict
               )

    data, models = parse_config(cfg)

    print(models)
    train_utils.train_models(cfg, data, models)
    wandb.finish()


if __name__ == '__main__':
    main()
