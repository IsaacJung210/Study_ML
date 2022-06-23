import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    c = max(x)
    a = x - c
    return np.exp(a)/(np.sum(np.exp(a)))
