from implementations import *

y, data, labels = load_csv_data("train.csv")

# which indices have low variance -> none
data = data_preprocessing(data)
print(data.shape)
std_deviations = np.std(data, axis=0)
indices_zero_var = find_low_variance(std_dev=std_deviations, threshold=0.1)
print(indices_zero_var)
# print(data)
# print(y)

