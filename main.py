import dotenv
import os
import numpy as np

from scipy.optimize import least_squares
import tensorflow as tf
from pandas import DataFrame, read_csv
from utils import denormalize_with_params, lm_loss, normalize_with_params

from sklearn.metrics import r2_score
    


dotenv.load_dotenv()

# Read csv dataset.
whole_dataframe: DataFrame= read_csv('HardnessVickersDataset.csv')

X = whole_dataframe.drop(columns=['Hardness_HV_'])
y = whole_dataframe[['Hardness_HV_']]

# Build segments, get 70% for training, 15% for validation and 15% for testing
train_size = int(0.7 * len(whole_dataframe))
val_size = int(0.15 * len(whole_dataframe))
test_size = len(whole_dataframe) - train_size - val_size


train_X, val_X, test_X = np.split(X, [train_size, train_size + val_size])
train_y, val_y, test_y = np.split(y, [train_size, train_size + val_size])

X_min, X_max = train_X.min(), train_X.max()
y_min, y_max = train_y.min(), train_y.max()

# Normalize data. with values between -1 and 1
# Formula: (x - min) / (max - min) * 2 - 1
# https://stats.stackexchange.com/a/70807

norm_train_data = normalize_with_params(train_X, X_min, X_max)
norm_val_data   = normalize_with_params(val_X,   X_min, X_max)
norm_test_data  = normalize_with_params(test_X,  X_min, X_max)

norm_train_target = normalize_with_params(train_y, y_min, y_max)
norm_val_target   = normalize_with_params(val_y,   y_min, y_max)
norm_test_target  = normalize_with_params(test_y,  y_min, y_max)


# Build Sequential Model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(norm_train_data.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
learning_rate = 1e-3

epochs = int(os.getenv("EPOCHS", 5))

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# Get initial model weights
initial_params = np.concatenate([w.flatten() for w in model.get_weights()])


result = least_squares(lm_loss, initial_params, args=(model, norm_train_data, norm_train_target), method='trf', verbose=2)

# model.compile(optimizer="adam", loss=loss_fn, metrics=["mae"])

optimized_weights = result.x

shapes = [w.shape for w in model.get_weights()]
sizes = [np.prod(s) for s in shapes]
split_indices = np.cumsum(sizes)[:-1]
weights = np.split(optimized_weights, split_indices)
weights = [w.reshape(s) for w, s in zip(weights, shapes)]
model.set_weights(weights)

predictions = model.predict(norm_test_data)

preds_real   = denormalize_with_params(predictions, y_min, y_max)
targets_real = denormalize_with_params(norm_test_target, y_min, y_max)

# Calculate R² score
r2 = r2_score(targets_real, preds_real)
print(f"R² score on test set: {r2:.4f}")