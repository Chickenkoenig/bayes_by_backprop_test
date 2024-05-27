import matplotlib.pyplot as plt
import wandb

def plot_dataset(data, title):
    x, y = data
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



#plot confidence intervalls
def plot_model_results(x_test, y_test, train_data, mean_values, std_values, clean_target_func, cfg):
    x_train, y_train = train_data
    plt.figure(figsize=(10, 8))
    plt.plot(x_test.data.numpy(), mean_values, color=cfg.plot.colors.mean, lw=3, label='Predicted Mean Model')
    plt.fill_between(x_test.data.numpy().T[0], mean_values - 3.0 * std_values, mean_values + 3.0 * std_values,
                     alpha=cfg.plot.alpha, color=cfg.plot.colors.confidence, label='99.7% confidence interval')
    plt.plot(x_train.data.numpy(), y_train.data.numpy(), '.', color=cfg.plot.colors.test_set, markersize=cfg.plot.markersize, label='Training set')
    plt.plot(x_test.data.numpy(), clean_target_func(x_test).data.numpy(), color=cfg.plot.colors.target_function, markersize=cfg.plot.markersize,
             label='Target function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')


    wandb.log({"Prediction Plot": wandb.Image(plt)})
    plt.show()




