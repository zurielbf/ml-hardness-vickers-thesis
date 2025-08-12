# import torch


from pandas import DataFrame, read_csv
from utils import normalize_data

# Read csv dataset.
whole_dataframe: DataFrame= read_csv('HardnessVickersDataset.csv')


# Build segments, get 70% for training, 15% for validation and 15% for testing
train_size = int(0.7 * len(whole_dataframe))
val_size = int(0.15 * len(whole_dataframe))
test_size = len(whole_dataframe) - train_size - val_size


# Normalize data. with values between -1 and 1
# Formula: (x - min) / (max - min) * 2 - 1
# https://stats.stackexchange.com/a/70807

norm_whole_df_no_target = normalize_data(whole_dataframe.drop(columns=['Hardness_HV_']))

norm_whole_df_target = normalize_data(whole_dataframe['Hardness_HV_'])

norm_train_data = norm_whole_df_no_target[:train_size]
norm_val_data = norm_whole_df_no_target[train_size:train_size + val_size]
norm_test_data = norm_whole_df_no_target[train_size + val_size:]

norm_train_target = norm_whole_df_target[:train_size]
norm_val_target = norm_whole_df_target[train_size:train_size + val_size]
norm_test_target = norm_whole_df_target[train_size + val_size:]

print(norm_train_data)
print(norm_val_data)
print(norm_test_data)
print(norm_train_target)
print(norm_val_target)
print(norm_test_target)