import src.data_generator as data_generator

def get_train_data(cfg):
    x_start = cfg.data.x_start
    x_end = cfg.data.x_end
    num_train_points = cfg.data.num_train_points
    func = getattr(data_generator, cfg.data.noisy_target)
    if cfg.data.gap:
        gap_start = cfg.data.gap_start
        gap_end = cfg.data.gap_end
        x, y = data_generator.generate_data_with_gaps(func, x_start, x_end, num_train_points, gap_start, gap_end)
    else:
        x, y = data_generator.generate_data(func, x_start, x_end, num_train_points)
    return x, y

def get_validation_data(cfg):
    x_start = cfg.data.x_start
    x_end = cfg.data.x_end
    num_train_points = cfg.data.num_validation_points
    func = getattr(data_generator, cfg.data.noisy_target)
    x, y = data_generator.generate_data(func, x_start, x_end, num_train_points)
    return x, y

def get_test_data(cfg):
    x_start = cfg.data.x_start
    x_end = cfg.data.x_end
    num_train_points = cfg.data.num_test_points
    func = getattr(data_generator, cfg.data.noisy_target)
    x, y = data_generator.generate_data(func, x_start, x_end, num_train_points)
    return x, y
