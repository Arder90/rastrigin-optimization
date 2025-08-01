import numpy as np

def rastrigin(X):
    x, y = X[0], X[1]
    return 20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))