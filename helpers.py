import numpy as np
import csv
import math as mp


def standardize(x):
    """
    Standardize each feature of the original data set (x)
    Arguments:
        - x: a data set of size (N,D) with the columns corresponding to the features

    Returns:
        - standardize_x: the standardized data
    """
    mean_x = np.mean(x, axis=0)
    standardized_x = x - mean_x
    std_x = np.std(x, axis=0)
    standardized_x = standardized_x / std_x
    return standardized_x


def find_low_variance(std_dev, threshold):
    """
    Function that finds the indices of the columns that have a low variance

    Arguments:
        - std_dev: numpy array of size (N,)
        - threshold: float

    Returns:
        - indices: the indices of the features where the standard deviation is lower than the threshold
    """
    indices = np.where(np.abs(std_dev) < threshold)
    return indices


def change_angle(data, indices):
    """
    Function that changes the features that are angles to their sine and cosine

    Arguments:
        - data: the data where the features are to be changed
        - indices: the indices of the columns to be changed

    Returns:
        - data: the data with the changed features
    """

    for i in indices:
        cosinus = np.zeros(data.shape[0])
        for j in range(data.shape[0]):
            cosinus[j] = mp.cos(data[j][i])
            data[j][i] = mp.sin(data[j][i])
        # hstack=concatenate with axis =1
        data = np.hstack((data, cosinus.reshape(-1, 1)))
    return data


def remove_outliers(data):
    """
    Change the value of the outliers present in the data to the median of the feature

    Arguments:
        - data: the data where we want the outliers to be changed

    Returns:
        - data: data where outliers have been changed to the median value of the column
    """

    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    boolean_matrix = np.logical_or(
        data > means + 3 * std_devs, data < means - 3 * std_devs
    )
    data[boolean_matrix] = np.NaN
    data = np.nan_to_num(data, nan=np.nanmedian(data, axis=0))

    return data


def remove_outliers_to_std(data):
    """
    Change the value of the outliers present in the data to +/- 3x the standard deviation of the feature

    Arguments:
        - data: the data where we want the outliers to be changed

    Returns:
        - data: data where outliers have been changed to the +/- 3x the standard deviation of the column
    """

    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    bigger_than_matrix = data > (means + 3 * std_devs)
    smaller_than_matrix = data < (means - 3 * std_devs)
    data[bigger_than_matrix] = np.NaN
    data = np.nan_to_num(data, nan=means + 3 * std_devs)
    data[smaller_than_matrix] = np.NaN
    data = np.nan_to_num(data, nan=means - 3 * std_devs)

    return data


def features_poly_extension(x, best_degrees):
    """
    Polynomial extension of the dataset where each feature is put to the degree giving the smallest error
    The first column is not extended since it's the offset (column of 1s)

    Arguments:
        - x: shape=(N+1,D), data set we want to extend with polynomials
        - best_degrees: list of degrees to put all the features we want to extend

    Returns:
        - x: the polynomially extended data set x
    """

    for i in range(len(best_degrees)):
        x = build_poly(x, best_degrees[i], col_to_expand=1)

    return x


def predict_labels(x, w):
    """
    Returns the predicted labels given data set x and weights w

    Arguments:
        - x: shape=(N,D), data set from which we want to predict our labels
        - w: shape=(D,), weights used to make prediction

    Returns:
        - prediction: numpy array of the predicted labels of dataset x
    """
    prediction = np.dot(x, w)
    # sets labels that are not equal to 1 or -1 to their closer number
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1
    return prediction


def compute_mse(y, tx, w):
    """
    Calculate the loss using MSE
    Arguments:
        - y: shape=(N, )
        - tx: shape=(N,D)
        - w: shape=(D,). The vector of model parameters.

    Returns:
        - a scalar, the value of the mse for the given parameters
    """
    e = y - tx.dot(w)

    return 1 / 2 * np.mean(e**2)


def load_csv_data(data_path, sub_sample=False):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)

    Arguments:
        - data_path: the path of the data to load

    Returns:
        - yb: numpy array with the labels
        - input_data: the data
        - ids: the ids of the dataset
    """
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

    Arguments:
        - ids: event ids associated with each prediction
        - y_pred: predicted class labels
        - name: string name of .csv output file to be created
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def compute_gradient(y, tx, w):
    """
    Computes the gradient at w.
    Arguments:
        - y: numpy array of shape=(N, )
        - tx: numpy array of shape=(N,D)
        - w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        - gradient: numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    n = y.shape[0]
    gradient = -(tx.T.dot(e)) / n
    return gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the mini batches

    Arguments:
        - y: numpy array of shape=(N, )
        - tx: numpy array of shape=(N,D)
        - batch_size: the size of the batch
        - num_batches: the number of batches, default value is
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
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    This function returns the matrix formed by applying the polynomial basis to the col_to_expand column of x

    Arguments:
        - x: numpy array of shape (N, D), N is the number of samples.
        - degree: integer.
        - col_to_expand: integer, the index of the column to which the polynomial basis function will be applied

    Returns:
        - data: numpy array of shape (N, D+degree)

    """
    poly = np.zeros((x.shape[0], degree))
    feature = x[:, col_to_expand]
    for j in range(1, degree + 1):
        poly[:, j - 1] = feature**j
    data = np.delete(x, col_to_expand, 1)
    data = np.append(data, poly, axis=1)

    return data


# LOGISTIC


def sigmoid(t):
    """
    Apply the sigmoid function on t.

    Arguments:
        - t: scalar or numpy array

    Returns:
        sigmoid_: scalar or numpy array to which the sigmoid function has been applied
    """
    sigmoid_ = 1.0 / (1 + np.exp(-t))
    return sigmoid_


def compute_mse_logistic(y, tx, w):
    """
    Compute the mse for error vector e for logistic regression
    Arguments:
        - y:  shape=(N, 1)
        - tx: shape=(N, D)
        - w:  shape=(D, 1)

    Returns:
        - scalar representing the MSE
    """
    y = np.reshape(y, (-1, 1))
    e = y - tx.dot(w)

    return 1 / 2 * np.mean(e**2)


def compute_loss_logistic(y, tx, w):
    """
    Compute the loss with the negative log likelihood

    Arguments:
        - y:  shape=(N, 1)
        - tx: shape=(N, D)
        - w:  shape=(D, 1)

    Returns:
        - the loss value, for the given parameters
    """

    y = np.reshape(y, (-1, 1))
    sig = sigmoid(tx.dot(w))
    loss = -y.T @ (np.log(sig)) - (1 - y).T @ (np.log(1 - sig))
    return np.squeeze(loss / y.shape[0])


def compute_gradient_logistic(y, tx, w):
    """
    Compute the gradient of the loss.

    Args:
        - y:  shape=(N, 1)
        - tx: shape=(N, D)
        - w:  shape=(D, 1)

    Returns:
        - a vector of shape (D, 1)
    """

    y = np.reshape(y, (-1, 1))
    sig = sigmoid(tx.dot(w))
    gradient = np.dot(tx.T, (sig - y))
    return gradient / y.shape[0]


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Returns the loss and the updated w.

    Arguments:
        - y:  shape=(N, 1)
        - tx: shape=(N, D)
        - w:  shape=(D, 1)
        - gamma: float, the step size

    Returns:
        - loss: scalar number
        - w: shape=(D, 1), the updated weights
    """

    gradient = compute_gradient_logistic(y, tx, w)
    loss = compute_loss_logistic(y, tx, w)
    w = w - gamma * gradient

    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Computes the penalized logistic regression's gradient and loss

    Arguments:
        - y:  shape=(N, 1)
        - tx: shape=(N, D)
        - w:  shape=(D, 1)
        - lambda_: scalar: teh penalization term

    Returns:
        - loss: scalar number
        - gradient: shape=(D, 1)
    """

    loss = compute_loss_logistic(y, tx, w)
    gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w

    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.

    Arguments:
        - y:  shape=(N, 1)
        - tx: shape=(N, D)
        - w:  shape=(D, 1)
        - gamma: scalar, the step size
        - lambda_: scalar, the penalization term

    Returns:
        - loss: scalar number, the loss
        - w: shape=(D, 1), the updates weights
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient

    return loss, w


def change_labels_to_zero(y):
    """
    Function to change the labels from {-1;1} to {0;1}

    Arguments:
        - y:  shape=(N, 1), numpy array of the labels to be changed

    Returns:
        - y_updated: a vector of shape (N, 1) with the updated labels
    """
    y_updated = np.ones(len(y))

    for i in range(len(y)):
        if y[i] <= 0:
            y_updated[i] = 0
    # y_updated[y <= 0] = 0

    return y_updated


def change_labels_to_minusone(y):
    """
    Function to change the labels from {0;1} to {-1;1}

    Arguments:
        - y:  shape=(N, 1), numpy array of the labels to be changed

    Returns:
        - - y_updated: a vector of shape (N, 1) with the updated labels
    """
    y_updated = np.ones(len(y))
    for i in range(len(y)):
        if y[i] <= 0.5:
            y_updated[i] = -1

    return y_updated


def predict_labels_logistic(x, w):
    """
    Returns the predicted labels given data set x and weights w for logistic regression

    Arguments:
        - x: shape=(N,D), data set from which we want to predict our labels
        - w: shape=(D,), weights used to make prediction

    Returns:
        - prediction: the predicted labels of dataset x
    """
    prediction = np.dot(x, w)
    # sets labels that are not equal to 1 or -1 to their closer number
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = -1
    return prediction
