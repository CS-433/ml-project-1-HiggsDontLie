from implementations import *


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
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
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


def cv_least_squares(y, x, k_fold, seed):
    """ performs "k_fold"-cross validation of the least_square method

        Args:
            y:          shape=(N,)
            x:          shape=(N,D)
            k_fold:     scalar, the number of times we will perform the cross-validation
            seed:       set the seed to have reproducible results

        Returns:
            mse_tr, mse_te: the train and test error found by averaging all the train and test errors of each fold
        """
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)
    mse_tr_temp = []
    mse_te_temp = []

    for k in range(k_fold):
        x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
        w, mse_tr_i = least_squares(y_train, x_train)
        mse_te_i = compute_mse(y_test, x_test, w)
        mse_tr_temp.append(mse_tr_i)
        mse_te_temp.append(mse_te_i)

    mse_tr = np.mean(mse_tr_temp)
    mse_te = np.mean(mse_te_temp)
    return mse_tr, mse_te


def cv_polynomial_reg(y, x, degree, col_to_expand, k_fold, seed):
    """ performs "k_fold"-cross validation of the polynomial regression method

            Args:
                y:          shape=(N,)
                x:          shape=(N,D)
                degree:     scalar, degree to which the feature need to be expanded
                col_to_expand scalar, which feature will be expanded
                k_fold:     scalar, the number of times we will perform the cross-validation
                seed:       set the seed to have reproducible results

            Returns:
                mse_tr, mse_te: the train and test error found by averaging all the train and test errors of each fold
            """
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)
    mse_tr_temp = []
    mse_te_temp = []

    for k in range(k_fold):
        x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
        w, mse_tr_i = polynomial_regression(y_train, x_train, degree, col_to_expand)
        x_test_poly = build_poly(x_test, degree, col_to_expand)
        mse_te_i = compute_mse(y_test, x_test_poly, w)
        mse_tr_temp.append(mse_tr_i)
        mse_te_temp.append(mse_te_i)

    mse_tr = np.mean(mse_tr_temp)
    mse_te = np.mean(mse_te_temp)
    return mse_tr, mse_te


def cv_gradient_des(y, x, gammas, k_fold, seed):
    """ performs "k_fold"-cross validation of the gradient descend method

                Args:
                    y:          shape=(N,)
                    x:          shape=(N,D)
                    gamma:      array of the different step sizes
                    k_fold:     scalar, the number of times we will perform the cross-validation
                    seed:       set the seed to have reproducible results

                Returns:
                    mse_train, mse_test:    dictionnary of errors associated with each gamma after averaging over
                                            k-fold cross-validation. The order in the array corresponds to the
                                            order the gammas were given in

                """
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)

    # For now we set these parameters as fixed values, as we aren't tuning them
    # if we wish to tune them we would need to enter them as arguments in the function
    max_iters = 100
    initial_w = np.zeros(x.shape[1])

    mse_train = []
    mse_test = []

    for gamma in gammas:
        mse_train_local = []
        mse_test_local = []
        for k in range(k_fold):
            x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
            weights, mse_train_i = mean_squared_error_gd(
                y_train, x_train, initial_w, max_iters, gamma
            )
            mse_test_i = compute_mse(y_test, x_test, weights)
            mse_train_local.append(mse_train_i)
            mse_test_local.append(mse_test_i)

        mse_tr = np.mean(mse_train_local)
        mse_te = np.mean(mse_test_local)

        mse_train.append(mse_tr)
        mse_test.append(mse_te)

    return mse_train, mse_test


def cv_ridge_reg(y, x, lambda_, k_fold, seed):
    """ performs "k_fold"-cross validation of the ridge regression method

            Args:
                y:          shape=(N,)
                x:          shape=(N,D)
                lambda_:     array, penalty applied on the weights
                k_fold:     scalar, the number of times we will perform the cross-validation
                seed:       set the seed to have reproducible results

            Returns:
                lambda_mse_tr, lambda_mse_te:   dictionary of associated train and test errors
                                                found by averaging all the train and test errors of
                                                each fold with their corresponding lambda
            """
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)

    lambda_mse_tr = {}
    lambda_mse_te = {}

    for lamb in lambda_:
        mse_tr_temp = []
        mse_te_temp = []
        for k in range(k_fold):
            x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
            w, mse_tr_i = ridge_regression(y_train, x_train, lamb)
            mse_te_i = compute_mse(y_test, x_test, w)
            mse_tr_temp.append(mse_tr_i)
            mse_te_temp.append(mse_te_i)

        mse_tr = np.mean(mse_tr_temp)
        mse_te = np.mean(mse_te_temp)
        lambda_mse_tr[lamb] = mse_tr
        lambda_mse_te[lamb] = mse_te

    return lambda_mse_tr, lambda_mse_te