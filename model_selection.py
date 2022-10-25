from cross_validation import *
from plots import *


# set seed to be able to reproduce our results
seed = 1
# 10-fold cross-validation (cv)
k_fold = 10
y, data, labels = load_csv_data("train.csv")
# we found no features with low variance, so indices_zero_var is an empty array
x = data_preprocessing(data)

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
