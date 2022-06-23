import numpy as np

def mse(y, t):     
    return np.sum((y-t)**2)/y.shape[0]

def cross_entropy_error(y,t):
    if 1 == y.ndim: 
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    delta = 1e-7
    return -np.sum(np.log(y+delta) * t)/y.shape[0]


# 교수님꺼 
# import numpy as np

# def mse(y,t):
#     return (1/2*np.sum((y-t)**2))*y.size

# def cross_entropy_err(y,t):
#     delta = 1e-7
#     return np.sum(-np.sum(t*np.log(y+delta)))/y.size