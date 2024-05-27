import src.config_parser.data_parser as data_parser
import src.config_parser.model_parser as model_parser
def parse_config(cfg):
    models = model_parser.get_models(cfg)
    data = data_parser.get_data(cfg)
    return data, models