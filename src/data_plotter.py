import matplotlib.pyplot as plt

def plot_dataset(data):
    x, y = data
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.title("Scatter plot of dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()