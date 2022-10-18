from cross_validation import *

# set seed to be able to reproduce our results
seed = 1

y, data, labels = load_csv_data("train.csv")
# we found no features with low variance, so indices_zero_var is an empty array
x = data_preprocessing(data)
# 7-fold cv of least square
mse_tr, mse_te = cv_least_squares(y, x, 7, seed)

# 0.574407
print(mse_tr)
# 0.574276
print(mse_te)




