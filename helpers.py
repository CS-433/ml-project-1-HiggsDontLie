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
    boolean_matrix = np.logical_or(
        data > means + 3 * std_devs, data < means - 3 * std_devs
    )
    data[boolean_matrix] = np.NaN
    data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))

    return data


def predict_labels(x, w):
    """returns the predicted labels given data set x and weights w

           Args:
               x: shape=(N,D), data set from which we want to predict our labels
               w: shape=(D,), weights used to make prediction

           Returns:
               the predicted labels of dataset x
           """
    prediction = np.dot(x, w)
    # sets labels that are not equal to 1 or -1 to their closer number
    prediction[prediction >= 0] = 1
    prediction[prediction < 0] = -1
    return prediction


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    y_pred = predict_labels(tx, w)
    return np.mean(np.power(y - y_pred, 2)) / 2


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


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