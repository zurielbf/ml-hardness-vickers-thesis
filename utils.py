
import numpy as np

def normalize_with_params(df, min_val, max_val):
    denom = (max_val - min_val)
    denom[denom == 0] = 1  # avoid division by zero
    return ((df - min_val) / denom) * 2 - 1

def denormalize_with_params(df, min_val, max_val):
    return ((df + 1) / 2) * (max_val - min_val) + min_val

def lm_loss(params, model, X, y):
    # Get the shapes of the model's weights
    shapes = [w.shape for w in model.get_weights()]
    sizes = [np.prod(s) for s in shapes]
    split_indices = np.cumsum(sizes)[:-1]
    # Split and reshape params
    weights = np.split(params, split_indices)
    weights = [w.reshape(s) for w, s in zip(weights, shapes)]
    model.set_weights(weights)
    y_pred = model(X, training=False)
    return (y_pred.numpy().flatten() - y.values.flatten())