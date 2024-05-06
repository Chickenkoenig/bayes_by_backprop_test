from src.models.bbb_model import BBBModel
from src.models.dropout_model import MCDropoutNet
def get_model(cfg):
    model_type = cfg.model.type
    if model_type == "bbb":
        return BBBModel(cfg)
    elif model_type == "dropout":
        return MCDropoutNet(cfg)