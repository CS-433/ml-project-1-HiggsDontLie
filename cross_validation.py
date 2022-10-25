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


def cv_polynomial_reg(y, x, degree, k_fold, seed, col_to_expand=-1):
    """ performs "k_fold"-cross validation of the polynomial regression method

            Args:
                y:          shape=(N,)
                x:          shape=(N,D)
                degree:     scalar, degree to which the feature need to be expanded
                col_to_expand list, which features will be expanded
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
        if col_to_expand == -1:
            for n in range(x_test.shape[1]):
                x_test = build_poly(x_test, degree, 0)
        else:
            x_test = build_poly(x_test, degree, col_to_expand)
        mse_te_i = compute_mse(y_test, x_test, w)
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
                    gammas:      array of the different step sizes
                    k_fold:     scalar, the number of times we will perform the cross-validation
                    seed:       set the seed to have reproducible results

                Returns:
                    mse_train, mse_test:    arrays of errors associated with each gamma after averaging over
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


def cv_stoch_gradient_des(y, x, gammas, k_fold, seed):
    """ performs "k_fold"-cross validation of the stochastic gradient descend method

                Args:
                    y:          shape=(N,)
                    x:          shape=(N,D)
                    gammas:      array of the different step sizes
                    k_fold:     scalar, the number of times we will perform the cross-validation
                    seed:       set the seed to have reproducible results

                Returns:
                    mse_train, mse_test:    arrays of errors associated with each gamma after averaging over
                                            k-fold cross-validation. The order in the array corresponds to the
                                            order the gammas were given in

                """
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)

    # For now we set these parameters as fixed values, as we aren't tuning them
    # if we wish to tune them we would need to enter them as arguments in the function
    max_iters = 200
    initial_w = np.zeros(x.shape[1])

    mse_train = []
    mse_test = []

    for gamma in gammas:
        mse_train_local = []
        mse_test_local = []
        for k in range(k_fold):
            x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
            weights, mse_train_i = mean_squared_error_sgd(
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
                mse_tr, mse_te:  train and test errors found by averaging all the train and test errors of each fold
            """
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)

    mse_tr = []
    mse_te = []

    for lamb in lambda_:
        mse_tr_temp = []
        mse_te_temp = []
        for k in range(k_fold):
            x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
            w, mse_tr_i = ridge_regression(y_train, x_train, lamb)
            mse_te_i = compute_mse(y_test, x_test, w)
            mse_tr_temp.append(mse_tr_i)
            mse_te_temp.append(mse_te_i)

        mse_tr.append(np.mean(mse_tr_temp))
        mse_te.append(np.mean(mse_te_temp))

    return mse_tr, mse_te


# function that tries to put each feature on a degree 1-10 and see which one gives better results
def find_best_degree(y, x):
    """ Elevates each features on a degree from 1 to 10
    Computes the test and train mse of the dataset with this particular extension using 10-fold cv
    Returns an array of the degrees giving the smallest test mse for each feature
    (degree 0 is the best degree of extension for feature 1, etc...)

                Args:
                    y:          shape=(N,), the labels of the training set
                    x:          shape=(N,D), the training set containing an offset (col of 1s) and 4 boolean features
                Returns: array containing the best degree for each feature excluding the dummy variables
                     """
    degrees = 10
    nb_col = x.shape[1]
    mse_tr = np.zeros((degrees, nb_col - 5))
    mse_te = np.zeros((degrees, nb_col - 5))
    # we do not try it on column 0 since it's a col full of 1s
    # we do not try the degrees of the last 4 columns since it's the dummy variables (i.e. only contains 1s or 0s)
    for c in np.arange(1, nb_col - 4):
        # we start at 1 because degree 0 is just columns of 1s
        for d in np.arange(1, degrees + 1):
            # 10-fold cv
            mse_tr_i, mse_te_i = cv_polynomial_reg(
                y, x, d, k_fold=10, seed=1, col_to_expand=c
            )
            mse_tr[d - 1, c - 1] = mse_tr_i
            mse_te[d - 1, c - 1] = mse_te_i
    # returns an array of the best degree for each feature (= row index of each column + 1 since we start with degree=1)
    return np.argmin(mse_te, axis=0) + 1


def cv_best_degrees_ridge(y, x, best_degrees, lambdas, k_fold, seed=1):
    """ performs "k_fold"-cross validation of the ridge regression method on a polynomial expansion of data x

                Args:
                    y:          shape=(N,)
                    x:          shape=(N,D) data set with an offset as the first column (col of 1s)
                    best_degrees: array, degrees to which each feature should be elevated
                    lambdas:     array, penalty applied on the weights
                    k_fold:     scalar, the number of times we will perform the cross-validation
                    seed:       set the seed to have reproducible results

                Returns:
                    mse_test, mse_train:  train and test errors found by averaging all the train and test errors of each fold
                """
    # expands the dataset by building a polynom of the degree giving the smallest mse for each feature
    x = features_poly_extension(x, best_degrees)
    mse_test, mse_train = cv_ridge_reg(y, x, lambdas, k_fold, seed)
    return mse_test, mse_train


def cv_poly_ridge(y, x, degrees, k_fold, lambdas, seed=1):
    """ performs "k_fold"-cross validation of the ridge regression method on different polynomial expansions of data x

                    Args:
                        y:          shape=(N,)
                        x:          shape=(N,D) data set with an offset as the first column (col of 1s)
                        degrees: array, different degrees to which all features should be elevated
                        k_fold:     scalar, the number of times we will perform the cross-validation
                        lambdas:    array, penalty applied on the weights
                        seed:       set the seed to have reproducible results

                    Returns:
                        smallest test mse found, the degree and lambda used to find this mse

                    """
    best_lambdas = []
    best_mses = []
    for d in degrees:
        x_poly = x
        for n in range(x.shape[1] - 4):
            x_poly = build_poly(x_poly, d, col_to_expand=1)
        mse_tr_i, mse_te_i = cv_ridge_reg(y, x_poly, lambdas, k_fold, seed)
        best_index = np.argmin(mse_te_i)
        best_lambdas.append(lambdas[best_index])
        best_mses.append(mse_te_i[best_index])

    ind_best_degree = np.argmin(best_mses)

    return np.min(best_mses), degrees[ind_best_degree], best_lambdas[ind_best_degree]
