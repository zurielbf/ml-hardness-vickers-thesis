
import numpy as np


def normalize_with_params(df, min_val, max_val):
    denom = (max_val - min_val)
    denom[denom == 0] = 1  # avoid division by zero
    return ((df - min_val) / denom) * 2 - 1

def denormalize_with_params(df, min_val, max_val):
    df = np.asarray(df)
    min_val = np.asarray(min_val)
    max_val = np.asarray(max_val)
    return ((df + 1) / 2) * (max_val - min_val) + min_val
