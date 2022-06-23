import numpy as np


def mse(y, t):
    return (1/2)*np.sum((y-t)**2)


def cross_entropy_err(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / y.shape[0]
