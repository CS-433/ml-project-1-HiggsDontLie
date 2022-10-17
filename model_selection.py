from implementations import *
from plots import *


data = load_data("train.csv")
# we found no features with low variance, so indices_zero_var is an empty array
y, x = data_preprocessing(data, [])

# 7-fold cross-validation of least_squares
k_indices = build_k_indices(y, k_fold=7, seed=1)
mse_tr_temp = []
mse_te_temp = []

for k in range(7):
    x_train, y_train, x_test, y_test = build_sets_cv(y, x, k_indices, k)
    w, mse_tr_i = least_squares(y_train, x_train)
    mse_te_i = compute_mse(y_test, x_test, w)
    mse_tr_temp.append(mse_tr_i)
    mse_te_temp.append(mse_te_i)

mse_tr = np.mean(mse_tr_temp)
mse_te = np.mean(mse_te_temp)

print(mse_tr)
print(mse_te)




