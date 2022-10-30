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
    """
    Visualization of the curves of the training and test mse
    Arguments:
        - mse_tr, mse_te: numpy arrays of size (x, ), the training and test errors for each x_values
        - x_values: numpy array of size (x, ), all the values for which the mses were calculated
                    e.g. the lambdas in regularization
        - x_label, title: string, label of the x-axis and title of the graph
        - x_log_scale: bool, decides if the x-axis is in log scale or not
        - save_figure: if true the figure is saved in the plot folder as mses_title.png
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
    if save_figure:
        plt.savefig(r"plots\mses_" + title + ".png")
    plt.show()


def features_degrees_visualization(
    mse, degrees, title="test mse of each feature at each degrees", save_figure=True
):
    """
    Plots the curves of the mse with respect to the degree of expansion for each feature
    Arguments:
        - mse: numpy arrays of size (len(degrees), D), array containing the test errors of each feature
            for each degree of expansion
        - degrees: list of all the degrees to which the features are expanded
        - title: string, title of the graph
        - save_figure: if true the figure is saved as _title.png
    """
    fig, axs = plt.subplots(6, 6, figsize=(35, 35))
    n = 0
    for ax in axs.flatten()[:-2]:
        ax.plot(degrees, mse[:, n], marker=".")
        ax.set_xlabel("degrees")
        ax.set_ylabel("mse")
        ax.set_title("feature " + str(n + 1))
        ax.grid(True)
        n = n + 1
    axs.flatten()[-2].axis("off")
    axs.flatten()[-1].axis("off")
    if save_figure:
        plt.savefig(r"plots\_" + title + ".png")
    plt.show()
