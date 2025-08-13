import random
import dotenv


import numpy as np

import tensorflow as tf

from deap import base, creator, tools, algorithms
from pandas import DataFrame, read_csv
from utils import denormalize_with_params, normalize_with_params

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


# learning_rate = 1e-3
# epochs = int(os.getenv("EPOCHS", 5))
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# # Get initial model weights
# predictions = model.predict(norm_test_data)
# preds_real   = denormalize_with_params(predictions, y_min, y_max)
# targets_real = denormalize_with_params(norm_test_target, y_min, y_max)

# # Genetic Algo

def eval_model(
    individual
    ):
    # individual = [num_units, dropout_rate, learning_rate]
    num_units = max(1, int(individual[0]))  # Ensure at least 1 unit
    dropout_rate = min(max(individual[1], 0.0), 0.5)  # Clamp between 0 and 0.5
    learning_rate = max(1e-6, float(individual[2]))   # Ensure positive learning rate
    
    # Build and train model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(norm_train_data.shape[1],)),
        tf.keras.layers.Dense(num_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_units // 2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='mae'
    )
    model.fit(
        norm_train_data, 
        norm_train_target, 
        epochs=100, 
        verbose=0,
        validation_data=(norm_val_data, norm_val_target),
    )
    val_loss = model.evaluate(norm_val_data, norm_val_target, verbose=0)
    return (val_loss,)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("num_units", random.randint, 16, 128)
toolbox.register("dropout_rate", random.uniform, 0.0, 0.5)
toolbox.register("learning_rate", random.uniform, 1e-4, 1e-2)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_units, toolbox.dropout_rate, toolbox.learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_model)


population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

## SELECT BEST INDS
best_ind = tools.selBest(population, k=1)[0]
best_ind_num_units = int(best_ind[0])
best_ind_dropout_rate = best_ind[1]
best_ind_learning_rate = best_ind[2]

# 2. Rebuild and train the model with best hyperparameters
best_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(norm_train_data.shape[1],)),
    tf.keras.layers.Dense(best_ind_num_units, activation='relu'),
    tf.keras.layers.Dropout(best_ind_dropout_rate),
    tf.keras.layers.Dense(best_ind_num_units // 2, activation='relu'),
    tf.keras.layers.Dense(1)
])
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_ind_learning_rate), loss='mae')
best_model.fit(
    norm_train_data, 
    norm_train_target, 
    epochs=100, 
    verbose=2,
    validation_data=(norm_val_data, norm_val_target),
)

# 3. Save the trained model
best_model.save("best_model.keras")


loaded_model = tf.keras.models.load_model("best_model.keras")
predictions = loaded_model.predict(norm_test_data)

denormalized_predictions = denormalize_with_params(predictions, y_min, y_max)
denormalized_targets = denormalize_with_params(norm_test_target, y_min, y_max)

# Calculate R² score
r2 = r2_score(denormalized_targets, denormalized_predictions)
print(f"R² score on test set: {r2:.4f}")