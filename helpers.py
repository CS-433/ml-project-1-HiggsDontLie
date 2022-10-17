import numpy as np
import csv


def standardize(x):
    """Standardize each feature of the original data set (x)
    Args:
        x: a data set of size (N,D) with the columns corresponding to the features

    Returns:
        standardize_x: the standardized data
    """
    mean_x = np.mean(x, axis=0)
    standardized_x = x - mean_x
    std_x = np.std(x, axis=0)
    standardized_x = standardized_x / std_x
    return standardized_x


def find_low_variance(std_dev, threshold):
    """Args:
    std_dev: numpy array of size (N,)
    threshold: float
    Returns:
         the indices of the standard deviations that are lower than the threshold
    """
    indices = np.where(np.abs(std_dev) < threshold)
    return indices


# TODO: test this function, it should work tho
def remove_outliers(data):

    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    boolean_matrix = np.logical_or(data > means + 3*std_devs, data < means - 3*std_devs)
    data[boolean_matrix] = np.NaN
    data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))

    return data


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean(np.power(y-np.dot(tx, w), 2))/2


def load_data(path_dataset):
    """Load data set given by "path_dataset" and first cleaning of the data.
    The integer column "PRI_jet_num" becomes a float column
    label "b" becomes 1 and label "s" 0
    """
    data = np.genfromtxt(path_dataset, delimiter=",", dtype=None, names=True,
                         converters={1: lambda x: 1 if b"b" in x else 0, 24: lambda x: float(x)})
    # return the data as a 2d array
    data = np.asarray(data.tolist())
    return data


def write_csv(data, name_model):
    namefile = 'Predictions '+name_model+'.csv'
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


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    N = y.shape[0]
    gradient = -(tx.T.dot(e)) / N
    return gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the mini batches
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree, col_to_expand):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
        this function returns the matrix formed by applying the polynomial basis to the col_to_expand column of x

    Args:
        x: numpy array of shape (N, D), N is the number of samples.
        degree: integer.
        col_to_expand: integer, the index of the column to which the polynomial basis fct will be applied

    Returns:
        data: numpy array of shape (N, D+degree)

    example: build_poly(np.array([[0, 0.5, 2], [1, 2, 3]]), 2, 1)
    array([[0.   2.   1.   0.5  0.25]
            [1.   3.   1.   2.   4.  ]])
    """

    poly = np.zeros((x.shape[0], degree + 1))
    feature = x[:, col_to_expand]
    for j in range(degree + 1):
        poly[:, j] = feature ** j
    data = np.delete(x, col_to_expand, 1)
    data = np.append(data, poly, axis=1)

    return data


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: the number of folds
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    example: build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_sets_cv(y, x, k_indices, k):
    """returns training and test sets for cross validation
        puts the kth subgroup in test sets and the rest in training sets

        Args:
            y:          shape=(N,)
            x:          shape=(N,D)
            k_indices:  2D array returned by build_k_indices()
            k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)

        Returns:
            x_train, y_train: the data and its corresponding labels that will be used for training in the cv
            x_test, y_test: the data and its corresponding labels that will be used for testing in the cv
        """

    x_test = x[k_indices[k], :]
    y_test = y[k_indices[k]]
    # np.arrange(k_indices.shape[0]) creates a list of the size of the number of folds
    # ~ means not -> we take all the indices in k_indices except the ones in the kth fold
    train_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indices = train_indices.reshape(-1)
    x_train = x[train_indices, :]
    y_train = y[train_indices]

    return x_train, y_train, x_test, y_test
