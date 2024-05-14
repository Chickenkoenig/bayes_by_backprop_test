from src.config_parser import data_parser
from src.data import data_normalizer


def prepare_data(cfg):
    train_data, val_data, test_data = data_parser.get_data(cfg)
    return train_data, val_data, test_data

def normalize_data(train_data, val_data):
    train_data_normalized, val_data_normalized, normalization_params = data_normalizer.normalize_data(train_data, val_data)
    return train_data_normalized, val_data_normalized, normalization_params