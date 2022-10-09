import numpy as np
import csv


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return 1/(2*len(y)) * np.sum(np.power(y-np.dot(tx,w),2))


def load_data():
    """Load training data set and first cleaning of the data. The integer column "PRI_jet_num" becomes a float column"""
    path_dataset = "train.csv"
    data = np.genfromtxt(path_dataset, delimiter=",", dtype=None, names=True,
                         converters={1: lambda x: 0 if b"b" in x else 1, 24: lambda x: float(x)})
    return data


def writecsv(data,namemodel):
    namefile='Predictions '+namemodel+'.csv'
    header = ['Id', 'Prediction']

    ''' example on what we need -> best to give directly Id link with prediction
    data = [
        ['Albania', 28748],
        ['Algeria', 2381741],
        ['American Samoa', 199],
        ['Andorra', 468],
        ['Angola', 1246700]
    ]'''

    with open(namefile, 'w', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar."""

    txy = tx.T.dot(y)
    xtx = tx.T.dot(tx)
    w = np.linalg.solve(xtx, txy)
    mse = compute_loss(y, tx, w)

    return w, mse
