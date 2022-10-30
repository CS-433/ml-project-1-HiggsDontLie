import numpy as np

from cross_validation import *
from implementations import *
from plots import *

#y_tr, x_tr, ids_train = load_csv_data("train.csv")
# x_tr = data_preprocessing(x_tr)
# #print(x_tr.shape[1])
# mins = find_best_degree(y_tr, x_tr)
# print(mins)
# mins without cos and sin
# mins0 = [8, 6, 9, 9, 9, 9, 2, 9, 1, 8, 9, 2, 9, 5, 6, 1, 2, 9, 6, 7, 1, 7, 2, 4, 4, 3, 9, 1, 3, 7] -> used for best submission so far
# mins = [9, 9, 10, 10, 10, 9, 3, 10, 2, 9, 10, 3, 9, 8, 4, 6, 2, 9, 6, 6, 4, 10, 3, 2, 10, 3, 7, 10, 4, 8]
# mins with angles transformed into sin and cos
# mins0 = [9, 8, 10, 9, 10, 8, 2, 10, 7, 9, 9, 3, 8, 5, 7, 9, 4, 5, 3, 8, 4, 3, 3, 7, 9, 2, 8, 7, 8, 8, 1, 1, 5, 3, 2, 8]
# mins = [9, 8, 10, 10, 9, 8, 2, 10, 7, 10, 9, 3, 8, 8, 6, 9, 5, 8, 10, 10, 4, 9, 3, 5, 4, 4, 10, 8, 9, 8, 4, 6, 8, 1, 7]
# mins with dummy variables
# mins = [10, 10, 10, 10, 10, 3, 10, 8, 10, 9, 3, 8, 5, 7, 7, 9, 9, 7, 7, 6, 8, 10, 4, 8, 3, 10, 1, 9, 1, 4, 10, 8, 1, 1]
# mins with dummy variable AND the initial column
# mins = [9, 6, 10, 9, 9, 10, 10, 10, 9, 10, 9, 3, 8, 6, 10, 7, 9, 3, 8, 7, 2, 6, 1, 3, 10, 8, 4, 10, 5, 9, 6, 5, 9, 9, 5]

# x_tr = features_poly_extension(x_tr, mins)

'''
def cv_best_degrees(y, x, k_fold, best_degrees, seed=1):
    mse_tr_temp = []
    mse_te_temp = []
    k_indices = build_k_indices(y, k_fold, seed)
    for i in range(len(best_degrees)):
        x = build_poly(x, best_degrees[i], col_to_expand=1)
    for k in range(k_fold):
        x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
        w, mse_train_i = least_squares(y_train, x_train)
        mse_test_i = compute_mse(y_test, x_test, w)
        mse_tr_temp.append(mse_train_i)
        mse_te_temp.append(mse_test_i)

    mse_tr = np.mean(mse_tr_temp)
    mse_te = np.mean(mse_te_temp)
    return mse_tr, mse_te '''


# mse_train, mse_test = cv_best_degrees(y_tr, x_tr, k_fold=10, best_degrees=mins)


# 0.373982 -> without sin and cos
# 0.372545 -> without sin and cos but with col of 1s
# 0.373159 -> with sin and cos
# 0.372112 -> with sin and cos + col of 1s
# 0.374429 -> with dummy variables + col of 1s
# 0.372338 -> with dummy variables + col of 1s + initial PRI_jet_num
# print(mse_train)
# 0.374586 -> without sin and cos
# 0.373288 -> without sin and cos but with col of 1s
# 0.374306 -> with sin and cos
# 0.372936 -> with sin and cos + col of 1s
# 0.375376 -> with dummy variables + col of 1s
# 0.373223 -> with dummy variables + col of 1s + initial PRI_jet_num
# print(mse_test)
'''
def cv_best_degrees(y, x, k_fold, best_degrees, seed=1):
    features_poly_extension(x, best_degrees)
    mse_tr, mse_te = cv_least_squares(y, x, k_fold, seed)
    return mse_tr, mse_te'''

'''
def cv_best_degrees_ridge(y, x, best_degrees, lambdas, k_fold, seed=1):
    # expands the dataset by building a polynom of the degree giving the smallest mse for each feature
    for i in range(len(best_degrees)):
        x = build_poly(x, best_degrees[i], col_to_expand=0)
    mse_test, mse_train = cv_ridge_reg(y, x, lambdas, k_fold, seed)
    return mse_test, mse_train '''

'''
lambdas = np.logspace(-10, 0, 11)
mse_te, mse_tr = cv_best_degrees_ridge(y_tr, x_tr, mins, lambdas, 10, 1)
minl_tr = min(mse_tr, key=mse_tr.get)
mse_tr_min = mse_tr[minl_tr]
minl_te = min(mse_te, key=mse_te.get)
mse_te_min = mse_te[minl_te]
# 0.0001
print(minl_te)
# 0.3724613
print(mse_te_min)
# 0.373312
print(mse_tr_min)

lists = sorted(mse_tr.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

listste = sorted(mse_te.items()) # sorted by key, return a list of tuples
xte, yte = zip(*listste)
plt.semilogx(x, y, color='b', marker='*', label="Train error")
plt.semilogx(xte, yte, color='r', marker='*', label="Test error")
plt.show() '''



# create submission
'''
# load and pre-process the training and test data
y_te, test_data, ids_test = load_csv_data("test.csv")
x_te = data_preprocessing(test_data)
x_te = features_poly_extension(x_te, mins)

# train the model on the training set
# weights, mse_tr = least_squares(y_tr, x_tr) #-> does not work since there is an infinity of solutions
txy = x_tr.T.dot(y_tr)
xtx = x_tr.T.dot(x_tr)
weights = np.linalg.lstsq(xtx, txy, rcond=None)[0]
# predict labels of our test set and create a submission file
y_te = predict_labels(x_te, weights)
create_csv_submission(ids_test, y_te, "pred_poly.csv")
'''

'''
# create sumbmission without dummy variables, with sin and cos and with col of 1 and poly extension
mins = [9, 8, 10, 10, 9, 8, 2, 10, 7, 10, 9, 3, 8, 8, 6, 9, 5, 8, 10, 10, 4, 9, 3, 5, 4, 4, 10, 8, 9, 8, 4, 6, 8, 1, 7]
y_te, test_data, ids_test = load_csv_data("test.csv")
y_tr, x_tr, ids_train = load_csv_data("train.csv")
x_te = data_preprocessing(test_data)
x_te = features_poly_extension(x_te, mins)
x_tr = data_preprocessing(x_tr)
x_tr = features_poly_extension(x_tr, mins)
# train the model on the training set
weights, mse_tr = least_squares(y_tr, x_tr)
# predict labels of our test set and create a submission file
y_te = predict_labels(x_te, weights)
create_csv_submission(ids_test, y_te, "pred_poly.csv")'''

# try ridge regression on the specific poly ext. + submission with lambda = 1e-05
'''mins = [9, 8, 10, 10, 9, 8, 2, 10, 7, 10, 9, 3, 8, 8, 6, 9, 5, 8, 10, 10, 4, 9, 3, 5, 4, 4, 10, 8, 9, 8, 4, 6, 8, 1, 7]
y_tr, x_tr, ids_train = load_csv_data("train.csv")
x_tr = data_preprocessing(x_tr)
x_tr = features_poly_extension(x_tr, mins)
lambda_ = np.logspace(-10, 0, 11)
lambda_mse_tr, lambda_mse_te = cv_ridge_reg(y_tr, x_tr, lambda_, k_fold=10, seed=1)
# find the min mse associate to a lambda for train and test sets
minl_tr = min(lambda_mse_tr, key=lambda_mse_tr.get)
# 0.0.37211022222222223, lambda=1e-08
print(minl_tr)
print(lambda_mse_tr[minl_tr])
minl_te = min(lambda_mse_te, key=lambda_mse_te.get)
# 0.37292, lambda=1e-05
print(minl_te)
print(lambda_mse_te[minl_te])

# visualisation of train and test errors of ridge regression in function of lambda
lists = sorted(lambda_mse_tr.items())  # sorted by key, return a list of tuples
x, y = zip(*lists)  # unpack a list of pairs into two tuples
listste = sorted(lambda_mse_te.items())  # sorted by key, return a list of tuples
xte, yte = zip(*listste)
mses_visualization(y, yte, x, xte, x_log_scale=True)'''

# BEST SUBMISSION SO FAR
'''
mins = [9, 8, 10, 10, 9, 8, 2, 10, 7, 10, 9, 3, 8, 8, 6, 9, 5, 8, 10, 10, 4, 9, 3, 5, 4, 4, 10, 8, 9, 8, 4, 6, 8, 1, 7]
y_te, test_data, ids_test = load_csv_data("test.csv")
y_tr, x_tr, ids_train = load_csv_data("train.csv")
x_te = data_preprocessing(test_data)
x_te = features_poly_extension(x_te, mins)
x_tr = data_preprocessing(x_tr)
x_tr = features_poly_extension(x_tr, mins)
w, mse = ridge_regression(y_tr, x_tr, lambda_=1e-05)
print(mse)
y_te = predict_labels(x_te, w)
create_csv_submission(ids_test, y_te, "pred_ridge_best_degrees.csv")'''

# cross validate degree and ridge at the same time
'''
def cv_best_degrees_ridge(y, x, degrees, k_fold, lambdas, seed=1):
    best_lambdas = []
    best_mses = []
    for d in degrees:
        print(d)
        x_poly = x
        for n in range(x.shape[1]-5):
            x_poly = build_poly(x_poly, d, col_to_expand=1)
        print(x_poly.shape)
        mse_tr_i, mse_te_i = cv_ridge_reg(y, x_poly, lambdas, k_fold, seed)
        print(mse_te_i)
        best_index = np.argmin(mse_te_i)
        best_lambdas.append(lambdas[best_index])
        best_mses.append(mse_te_i[best_index])

    ind_best_degree = np.argmin(best_mses)
    print(best_lambdas)
    print(best_mses)

    return np.min(best_mses), degrees[ind_best_degree], best_lambdas[ind_best_degree]


y_te, test_data, ids_test = load_csv_data("test.csv")
y_tr, x_tr, ids_train = load_csv_data("train.csv")
x_te = data_preprocessing(test_data)
x_tr = data_preprocessing(x_tr)
degrees = range(1, 10)
lambdas = np.logspace(-10, 0, 11)
mse_te, best_degree, best_lambda = cv_best_degrees_ridge(y_tr, x_tr, degrees=degrees, k_fold=10, lambdas=lambdas)
# 0.37288 -> with dummy var
# 0.37288 -> without dummy var
print(mse_te)
# 9 -> with and without dummy var
print(best_degree)
# 0.0001 -> with and without dummy var
print(best_lambda)'''

# submission of degree 9 with lambda = 0.0001 ALSO BEST MODEL
y_te, test_data, ids_test = load_csv_data("test.csv")
y_tr, x_tr, ids_train = load_csv_data("train.csv")
x_te = data_preprocessing(test_data)
x_tr = data_preprocessing(x_tr)
for n in range(x_tr.shape[1] - 5):
    x_tr = build_poly(x_tr, 9, col_to_expand=1)
    x_te = build_poly(x_te, 9, col_to_expand=1)
w, mse = ridge_regression(y_tr, x_tr, lambda_=0.0001)
print(mse)
y_te = predict_labels(x_te, w)
create_csv_submission(ids_test, y_te, "pred_ridge0001_degree9.csv")
