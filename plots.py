import matplotlib.pyplot as plt


def mses_visualization(mse_tr, mse_te, x_values, x_label, title="training and test error",
                       x_log_scale=False, save_figure=True):
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
        plt.semilogx(x_values, mse_tr, marker=".", color='b', label='train error')
        plt.semilogx(x_values, mse_te, marker=".", color='r', label='test error')
    else:
        plt.plot(x_values, mse_tr, marker=".", color='b', label='train error')
        plt.plot(x_values, mse_te, marker=".", color='r', label='test error')

    plt.xlabel(x_label)
    plt.ylabel("mse")
    plt.title(title)
    plt.legend(loc=2)
    plt.grid(True)
    if save_figure:
        plt.savefig("mses_"+title+".png")
