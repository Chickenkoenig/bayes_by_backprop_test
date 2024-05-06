import matplotlib.pyplot as plt
import wandb

def plot_dataset(data):
    x, y = data
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.title("Scatter plot of dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#plot confidence intervalls
def plot_model_results(x_test, y_test, mean_values, std_values, clean_target_func, cfg):
    plt.figure(figsize=(10, 8))
    plt.plot(x_test.data.numpy(), mean_values, color=cfg.plot.colors.mean, lw=3, label='Predicted Mean Model')
    plt.fill_between(x_test.data.numpy().T[0], mean_values - 3.0 * std_values, mean_values + 3.0 * std_values,
                     alpha=cfg.plot.alpha, color=cfg.plot.colors.confidence, label='99.7% confidence interval')
    plt.plot(x_test.data.numpy(), y_test.data.numpy(), '.', color=cfg.plot.colors.test_set, markersize=cfg.plot.markersize, label='Test set')
    plt.plot(x_test.data.numpy(), clean_target_func(x_test).data.numpy(), color=cfg.plot.colors.target_function, markersize=cfg.plot.markersize,
             label='Target function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    wandb.log({"Prediction Plot": wandb.Image(plt)})

    plt.show()

