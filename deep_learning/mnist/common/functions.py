import numpy as np



def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def mean_square_error(y,t):
    error = y-t
    output = 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7 # To avoid nan
    return -np.sum(t * np.log(y+delta))
