import numpy as np

from cross_validation import *
from plots import *
from helpers import *
import black


# set seed to be able to reproduce our results
seed = 1
# 10-fold cross-validation (cv)
k_fold = 10
y, data, labels = load_csv_data("train.csv")
# we found no features with low variance, so indices_zero_var is an empty array
x = data_preprocessing(data)

# compute mse on train and test sets for logistic regression after cross validation:

# set seed to be able to reproduce our results
# seed = 1
# 10-fold cross-validation (cv)
# k_fold = 10
# y, data, labels = load_csv_data("train.csv")
# we found no features with low variance, so indices_zero_var is an empty array
# x = data_preprocessing(data)
# initial_w = np.zeros(x.shape[1])
# max_iters = 100
# gammas = np.logspace(-5, -1, 15)

# gamma_mse_train, gamma_mse_test = cv_logistic_regression(y, x, gammas, k_fold, seed)
# mses_visualization(gamma_mse_train, gamma_mse_test, gammas, 'gamma', x_log_scale=True, title="train and test error LR")

# min_error_train = min(gamma_mse_train)
# min_error_test = min(gamma_mse_test)

# min_gamma_train = gammas[np.where(gamma_mse_train == min_error_train)]
# min_gamma_test = gammas[np.where(gamma_mse_test == min_error_test)]

# print("Min gamma train:", min_gamma_train)
# Min gamma train: [0.01389495]
# print("Min error train:", min_error_train)
# Min error train: 0.5681829093377588

# print("Min gamma test:", min_gamma_test)
# Min gamma test: [0.00372759]
# print("Min error test:", min_error_test)
# Min error test: 0.39531978100633747



# compute mse on train and test sets for least squares after cv
# mse_tr, mse_te = cv_least_squares(y, x, 7, seed)

# 0.574407
# print(mse_tr)
# 0.574276
# print(mse_te)

# compute mse on train and test sets for gradient descend
# gammas = np.logspace(start=-1,stop=0,num=11)
# gamma_mse_tr, gamma_mse_te = cv_gradient_des(y,x,gammas,k_fold, seed)
# min_error_train = min(gamma_mse_tr)
# min_error_test = min(gamma_mse_te)
# mses_visualization(gamma_mse_tr, gamma_mse_te, gammas, 'gamma', x_log_scale=True, title="training and test error GD")

# min_gamma_train = gammas[np.where(gamma_mse_tr == min_error_train)]
# min_gamma_test = gammas[np.where(gamma_mse_te == min_error_test)]

# print(min_gamma_train)
# 0.31622777
# print(min_error_train)
# 0.5743733333333333
# print(min_gamma_test)
# 0.25118864
# print(min_error_test)
# 0.5745440000000002

# compute mse on train and test sets for stochastic gradient descend
# gammas = np.logspace(start=-6,stop=-1,num=11)
# gamma_mse_tr, gamma_mse_te = cv_stoch_gradient_des(y, x, gammas, k_fold, seed)

# mses_visualization(gamma_mse_tr, gamma_mse_te, gammas, 'gamma', x_log_scale=True, title="training and test error Stoch GD")

# min_error_train = min(gamma_mse_tr)
# min_error_test = min(gamma_mse_te)

# min_gamma_train = gammas[np.where(gamma_mse_tr == min_error_train)]
# min_gamma_test = gammas[np.where(gamma_mse_te == min_error_test)]

# print(min_gamma_train)
# 0.001
# print(min_error_train)
# 0.6287573333333334
# print(min_gamma_test)
# 0.001
# print(min_error_test)
# 0.6295520000000001

# (test)compute mse on train and test sets for polynomial regression with degree 12 after cv
# print(cv_polynomial_reg(y, x, 12, 2, k_fold, seed))

# compute mse on train and test sets for ridge regression after cv
# lambda_ = np.logspace(-5, 0, 15)
# lambda_mse_tr, lambda_mse_te = cv_ridge_reg(y, x, lambda_, k_fold, seed)

# find the min mse associate to a lambda for train and test sets

# minl_tr = min(lambda_mse_tr, key=lambda_mse_tr.get)
# 0.5723857777777778, lambda=0.00011787686347935866
# print(minl_tr)
# print(lambda_mse_tr[minl_tr])

# minl_te = min(lambda_mse_te, key=lambda_mse_te.get)
# 0.5725199999999999, lambda=2.2758459260747865e-05
# print(minl_te)
# print(lambda_mse_te[minl_te])


# visualisation of train and test errors of ridge regression in function of lambda
# lists = sorted(lambda_mse_tr.items())  # sorted by key, return a list of tuples
# x, y = zip(*lists)  # unpack a list of pairs into two tuples
# listste = sorted(lambda_mse_te.items())  # sorted by key, return a list of tuples
# xte, yte = zip(*listste)
# mses_visualization(y,yte, x, xte, x_log_scale=True)

"""
# find the minimal degree of each feature
# this can take a long time to run, here are a few mins already computed
best_degrees1 = [9, 8, 10, 10, 9, 8, 2, 10, 7, 10, 9, 3, 8, 8, 6, 9, 5, 8, 10, 10, 4, 9, 3, 5, 4, 4, 10, 8, 9, 8, 4, 6, 8, 1, 7]
# -> without dummy variable pre-processing, best submission so far
best_degrees2 = [9, 6, 10, 9, 9, 10, 10, 10, 9, 10, 9, 3, 8, 6, 10, 7, 9, 3, 8, 7, 2, 6, 1, 3, 10, 8, 4, 10, 5, 9, 6, 5, 9, 9, 5]
# -> mins with PRI_jet_num as a dummy variable
best_degrees = find_best_degree(y, x)
print(best_degrees)
x = features_poly_extension(x, best_degrees2)
mse_tr, mse_te = cv_least_squares(y, x, k_fold, seed)
# 0.372112 -> with best_degrees1
# 0.372338 -> with best_degrees2
print(mse_tr)
# 0.372936 -> with best_degrees1
# 0.373223 -> with best_degrees2
print(mse_te)"""

"""
# try the best degree method with ridge regression
lambdas = np.logspace(-10, 0, 11)
x = data_preprocessing(data)
best_degrees = [9, 6, 10, 9, 9, 10, 10, 10, 9, 10, 9, 3, 8, 6, 10, 7, 9, 3, 8, 7, 2, 6, 1, 3, 10, 8, 4, 10, 5, 9, 6, 5, 9, 9, 5]
mse_tr, mse_te = cv_best_degrees_ridge(y, x, best_degrees, lambdas, k_fold, seed)
min_mse_te = np.min(mse_te)
min_lambda_te = lambdas[np.argmin(mse_te)]
min_mse_tr = np.min(mse_tr)
min_lambda_tr = lambdas[np.argmin(mse_tr)]
# 0.0001
print(min_lambda_te)
# 0.372311
print(min_mse_tr)
# 0.373135
print(min_mse_te)
mses_visualization(mse_tr, mse_te, lambdas, "lambdas", title="cv ridge regression best degrees", x_log_scale=True)
"""

# cross validation of lambda and degree at the same time to find optimal degree and lambda
x = data_preprocessing(data)
degrees = range(1, 10)
lambdas = np.logspace(-10, 0, 11)
mse_te, best_degree, best_lambda = cv_poly_ridge(
    y, x, degrees=degrees, k_fold=10, lambdas=lambdas
)
# 0.372940 -> with mean
# 0.367328 -> with median for -999 and outliers
print(mse_te)
# 9
print(best_degree)
# 0.0001 -> with mean
# 1e-07 -> median for -999 and outlier
print(best_lambda)
