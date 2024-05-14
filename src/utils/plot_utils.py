from src.data import data_plotter, data_generator


def plot_data(data, title):
    data_plotter.plot_dataset(data, title)

def plot_results(test_data, train_data, mean_values, std_values, cfg):
    data_plotter.plot_model_results(test_data[0], test_data[1], train_data, mean_values, std_values, getattr(data_generator, cfg.data.ground_truth), cfg)