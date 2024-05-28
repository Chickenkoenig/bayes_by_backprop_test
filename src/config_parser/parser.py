import src.config_parser.data_parser as data_parser
import src.config_parser.model_parser as model_parser
import numpy as np
import torch
import random

def parse_config(cfg):
    initialize_seed(cfg)
    models = model_parser.get_models(cfg)
    data = data_parser.get_data(cfg)
    return data, models

def initialize_seed(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)