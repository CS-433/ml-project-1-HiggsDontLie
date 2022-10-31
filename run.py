from implementations import *
import black

# load and pre-process the training and test data
y_train, train_data, ids_train = load_csv_data("train.csv")
x_train = data_preprocessing_improved(train_data, col_to_remove=[15, 18, 20])
y_test, test_data, ids_test = load_csv_data("test.csv")
x_test = data_preprocessing_improved(test_data, col_to_remove=[15, 18, 20])
# polynomial expansion of all features except the column of 1s and the dummy variables
for n in range(x_train.shape[1] - 4):
    x_train = build_poly(x_train, 10, col_to_expand=1)
    x_test = build_poly(x_test, 10, col_to_expand=1)
# create the model
weights, mse = ridge_regression(y_train, x_train, lambda_=1e-07)
y_test = predict_labels(x_test, weights)
# create the submission file
create_csv_submission(ids_test, y_test, "pred_ridge_e7_degree10_pre-processing2.csv")
