from implementations import *
import black

# load and pre-process the training and test data
y_train, train_data, ids_train = load_csv_data("train.csv")
x_train = data_preprocessing_improved(train_data, indices_zero_var=[15, 18, 20])
y_test, test_data, ids_test = load_csv_data("test.csv")
x_test = data_preprocessing_improved(test_data, indices_zero_var=[15, 18, 20])
# polynomial expansion of all features except the column of 1s and the dummy variables
for n in range(x_train.shape[1] - 4):
    x_train = build_poly(x_train, 10, col_to_expand=1)
    x_test = build_poly(x_test, 10, col_to_expand=1)
# create the model
weights, mse = ridge_regression(y_train, x_train, lambda_=1e-07)
y_test = predict_labels(x_test, weights)
# create the submission file
create_csv_submission(ids_test, y_test, "pred_ridge_e7_degree10_pre-processing2.csv")

"""
# train the model on the training set
# weights, mse_train = least_squares(yin, x_train)
# weights, mse_train = ridge_regression(y_train, x_train, 0.1)
for n in range(x_train.shape[1] - 4):
    x_train = build_poly(x_train, 9, col_to_expand=1)
    x_test = build_poly(x_test, 9, col_to_expand=1)
# weights, mse = ridge_regression(y_train, x_train, lambda_=0.0001)
weights, mse = ridge_regression(y_train, x_train, lambda_=1e-7)
# predict labels of our test set and create a submission file
y_test = predict_labels(x_test, weights)
# create_csv_submission(ids_test, y_test, "pred_least_squares.csv")
# create_csv_submission(ids_test, y_test, "pred_ridge_regression.csv")
# create_csv_submission(ids_test, y_test, "pred_ridge0001_degree9.csv")
create_csv_submission(ids_test, y_test, "pred_ridge_e7_degree9_median.csv")  # best model so far """
