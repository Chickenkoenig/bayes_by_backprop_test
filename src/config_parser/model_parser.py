
from src.models.mlp_model import MLPModel
from src.models.bbb_model import BBBModel
from src.models.dropout_model import MCDropoutNet, MCDropoutLayer
from src.models.swag_model import SwagModel
import torch.nn as nn
import torchbnn as bnn

def get_models(cfg):
    models = []
    if cfg.model.type == "single":
        models.append(parse_model(cfg))
    else:
        for i in range(cfg.model.ensemble_type.network_count):
            models.append(parse_model(cfg))
    return models


def parse_model(cfg):
    model_arch = cfg.model.architecture.type
    layers = parse_model_layers(cfg)
    if model_arch == "bbb":
        return BBBModel(cfg, layers)
    elif model_arch == "dropout":
        return MCDropoutNet(cfg, layers)
    elif model_arch == "swag":
        return SwagModel(cfg, layers)
    elif model_arch == "mlp":
        return MLPModel(layers)

def parse_model_layers(cfg):
    layers = nn.ModuleList()
    for layer_cfg in cfg.model.architecture.layers:
        if layer_cfg.type == 'Linear':
            layer = nn.Linear(
                in_features=layer_cfg.in_features,
                out_features=layer_cfg.out_features
            )
        elif layer_cfg.type == 'ReLU':
            layer = nn.ReLU()
        elif layer_cfg.type == 'MCDropoutLayer':
            layer = MCDropoutLayer(cfg.model.architecture.dropout_prob)
        elif layer_cfg.type == 'BayesLinear':
            layer = bnn.BayesLinear(
                prior_mu=layer_cfg.prior_mu,
                prior_sigma=layer_cfg.prior_sigma,
                in_features=layer_cfg.in_features,
                out_features=layer_cfg.out_features
            )
        else:
            raise ValueError(f"Unsupported layer type: {layer_cfg.type}")
        layers.append(layer)

    return layers
