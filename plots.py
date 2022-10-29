import matplotlib.pyplot as plt
import black


def mses_visualization(
    mse_tr,
    mse_te,
    x_values,
    x_label,
    title="training and test error",
    x_log_scale=False,
    save_figure=True,
):
    """visualization the curves of the training and test mse
    Args
        mse_tr, mse_te: numpy arrays of size (x, ), the training and test errors for each x_values
        x_values: numpy array of size (x, ), all the values for which the mses were calculated
                    e.g. the lambdas in regularization
        x_label, title: string, label of the x-axis and title of the graph
        x_log_scale: bool, decides if the x-axis has a log scale or not
        save_figure: if true the figure is saved as mses_title.png

    """
    if x_log_scale:
        plt.semilogx(x_values, mse_tr, marker=".", color="b", label="train error")
        plt.semilogx(x_values, mse_te, marker=".", color="r", label="test error")
    else:
        plt.plot(x_values, mse_tr, marker=".", color="b", label="train error")
        plt.plot(x_values, mse_te, marker=".", color="r", label="test error")

    plt.xlabel(x_label)
    plt.ylabel("mse")
    plt.title(title)
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
    if save_figure:
        plt.savefig("mses_" + title + ".png")


def features_degrees_visualization(
    mse, degrees, title="test mse of each feature at each degrees", save_figure=True
):
    """plots the curves of the mse depending on the degree of expansion for each feature
    Creates a plot with one subplot for each feature
        Args
            mse: numpy arrays of size (len(degrees), D), array containing the test errors of each feature
                for each degree of expansion
            degrees: list of all the degrees to which the features are expanded
            title: string, title of the graph
            save_figure: if true the figure is saved as title.png

    """
    fig, axs = plt.subplots(6, 6, figsize=(35, 35))
    n = 0
    for i in range(6):
        for j in range(6):
            axs[i, j].plot(degrees, mse[:, n], marker=".")
            axs[i, j].set_xlabel("degrees")
            axs[i, j].set_ylabel("mse")
            axs[i, j].set_title("feature " + str(n + 1))
            axs[i, j].grid(True)
            n = n + 1
    plt.show()
    if save_figure:
        plt.savefig(title + ".png")
