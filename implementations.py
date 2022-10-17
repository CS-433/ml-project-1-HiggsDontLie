
import numpy as np
import csv


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def data_preprocessing(data):

    # this function need to standardize the data, remove -990 data points (replaced by mean of column)
    # remove features which have standard deviation of approx. 0, remove first column (identifiers)
    # and place Ys in a separate matrix

    N = data.np.shape([0])
    y = data[:1]
    # removing identifiers + Ys
    data = np.delete(data,0,1)
    data = np.delete(data,1,1)
    D = data.np.shape([1])

    #standardize the data:
    data, means, std_deviations = standardize(data)

    #remove features where st deviation is close to 0
    threshold = 1e-5
    indices_zero_var = np.where(np.logical_and(std_deviations < threshold, std_deviations > -threshold))
    data = np.delete(data,indices_zero_var,1 )

    #find -999 values
    boolean_matrix = (data ==-999)
    indices = np.where(boolean_matrix==1)




    return y, prepocessed_data

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


def load_data(path):
    """Load training data set and first cleaning of the data. The integer column "PRI_jet_num" becomes a float column"""
    path_dataset = path
    data = np.genfromtxt(path_dataset, delimiter=",", dtype=None, names=True,
                         converters={1: lambda x: 1 if b"b" in x else 0, 24: lambda x: float(x)})
    #return the data as a 2d array
    return np.array(data.tolist())


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


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features."""

    lamb = lambda_ * 2 * tx.shape[1]
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)