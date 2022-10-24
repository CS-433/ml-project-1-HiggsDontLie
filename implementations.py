import numpy as np

from helpers import *
import math as mp


def data_preprocessing(data, indices_zero_var=[]):

    # this function need to standardize the data, remove -999 data points (replaced by mean of column)
    # remove features which have standard deviation of approx. 0

    # find -999 values
    boolean_matrix = data == -999
    data[boolean_matrix] = np.NaN
    data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))

    # remove outliers
    data = remove_outliers(data)

    # change angle with their sinus and cosinus to keep the neighbourhood relationships, it concerns
    # the features: DER_met_phi_centrality, PRI_tau_phi, PRI_lep_phi, PRI_met_phi, PRI_jet_leading_phi
    # and PRI_jet_subleading_phi
    indices = [15, 18, 20, 25, 28]
    for i in indices:
        cosinus = np.zeros(250000)
        for j in range(250000):
            cosinus[j] = mp.cos(data[j][i])
            data[j][i] = mp.sin(data[j][i])
        # hstack=concatenate with axis =1
        data = np.hstack((data, cosinus.reshape(-1, 1)))

    # need to get the values of PRI_jet_num before standardization
    jet_zero = (data[:, 22] == 0).astype(float)
    jet_one = (data[:, 22] == 1).astype(float)
    jet_two = (data[:, 22] == 2).astype(float)
    jet_three = (data[:, 22] == 3).astype(float)

    # standardize the data
    data = standardize(data)
    
    # change the categorical feature PRI_jet_num into dummy variables
    data = np.concatenate((data, jet_zero.reshape(-1, 1), jet_one.reshape(-1, 1), jet_two.reshape(-1, 1),
                        jet_three.reshape(-1, 1)), axis=1)
    data = np.delete(data, 22, axis=1)

    # adds a row of 1 so that we can have an offset
    # TODO: find out why this causes problems in poly regression
    # data = np.c_[np.ones(len(data)), data]

    # remove features where st deviation is close to 0
    data = np.delete(data, indices_zero_var, 1)

    return data


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: the value of the loss (a scalar), corresponding to the input parameters w"""

    txy = tx.T.dot(y)
    xtx = tx.T.dot(tx)
    w = np.linalg.solve(xtx, txy)
    mse = compute_mse(y, tx, w)

    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: the value of the loss (a scalar), corresponding to the input parameters w"""

    lamb = lambda_ * 2 * tx.shape[0]
    a = tx.T.dot(tx) + (lamb * np.identity(tx.shape[1]))
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)

    return w, mse


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step size

    Returns:
        loss: the loss value (scalar) of the last iteration of GD
        w: numpy arrays of shape (D, ) containing the model parameters from the last iteration of GD
    """

    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
    # compute loss
    loss = compute_mse(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing stochastic gradient
                    default is 1
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the step size

    Returns:
        loss: the loss value (scalar) of the last iteration of SGD
        w: numpy arrays of shape (D, ) containing the model parameters from the last iteration of SGD
    """
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient and loss
            gradient = compute_gradient(y_batch, tx_batch, w)
            # update w
            w = w - gamma * gradient

    loss = compute_mse(y, tx, w)

    return w, loss


def polynomial_regression(y, tx, degree, col_to_expand):
    """Constructing the polynomial basis function expansion of the col_to_expand column of the data,
       and then running least squares regression.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        degree: integer
        col_to_expand: integer, feature that is expanded into polynomials

    Returns:
        weights: numpy array of shape=(degree+1), optimal weights for each feature expansion computed with least_square
        mse: scalar, the loss corresponding to the computed weights"""
    data = build_poly(tx, degree, col_to_expand)
    weights, mse = least_squares(y, data)

    return weights, mse
