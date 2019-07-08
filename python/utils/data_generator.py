import numpy as np

def generate_data_from_func(func, instance_size, feature_size):
    Xs = []
    ys = []
    for _ in range(instance_size):
        X, y = func(feature_size)
        Xs.append(X)
        ys.append(y)

    return np.array(Xs), np.array(ys)