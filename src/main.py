import numpy as np
import matplotlib.pyplot as plt
import model
import data_plotter
import data_generator

def main():
    data = data_generator.generate_data(data_generator.noisy_target, -2, 2, 500)
    data_plotter.plot_dataset(data)
    bnn_model = model.get_model()
    model.train_model(bnn_model, data, 2000, 0.01, 0.01)

    x_test, y_test = data_generator.generate_data(data_generator.noisy_target, -2, 2, 300)


    models_result = np.array([bnn_model(x_test).data.numpy() for k in range(10000)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T
    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])

    #plot confidence intervalls
    plt.figure(figsize=(10, 8))
    plt.plot(x_test.data.numpy(), mean_values, color='navy', lw=3, label='Predicted Mean Model')
    plt.fill_between(x_test.data.numpy().T[0], mean_values - 3.0 * std_values, mean_values + 3.0 * std_values,
                     alpha=0.2, color='navy', label='99.7% confidence interval')
    # plt.plot(x_test.data.numpy(),mean_values,color='darkorange')
    plt.plot(x_test.data.numpy(), y_test.data.numpy(), '.', color='darkorange', markersize=4, label='Test set')
    plt.plot(x_test.data.numpy(), data_generator.clean_target(x_test).data.numpy(), color='green', markersize=4,
             label='Target function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main()