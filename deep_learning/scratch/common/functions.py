import numpy as np



def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def mean_square_error(y,t):
    error = y-t
    output = 0.5*np.sum((y-t)**2)

# one_hot_label
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    if t.size == y.size:
        t = np.argmax(t, axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def ex_f(x):
    return np.sum(x**2)

def numerical_gradient(f,x):
    #f(x1+h)-f(x1-h)/2h,.....
    grad = np.zeros_like(x)
    step_size = 1e-4
    for i in range(x.shape[0]):
        if x.ndim ==1:
            h = np.zeros_like(x)
            h[i] += step_size
            tmp = x[i]
            #f(x+h)
            grad[i] = ((f(tmp+h)-f(tmp-h))/2*step_size)
        else:
            for j in range(x.shape[1]):
                h = np.zeros_like(x)
                h[i,j] += step_size
                tmp = x[i]
                #f(x+h)
                grad[i][j] = ((f(tmp+h)-f(tmp-h))/2*step_size)
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    # x = x - lr*grad(f)
    for i in range(step_num):
        gradient = numerical_gradient(f,x)
        x -= lr*gradient

    return x
