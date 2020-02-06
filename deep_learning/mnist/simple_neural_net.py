import sys, os
sys.path.append('c:\\users\\sungdonghyun\\rl\\ReinforcementLearning_study\\deep_learning\\deep-learning-from-scratch')
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle
from PIL import Image

def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))
def get_data():
    (x_train,t_train), (x_test, t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test
# 2 hidden layers with 784 input(number of pixels) and 10 output(0~9 digit probability)
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f) # pre trained network

    return network

def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0
# 1 batch
for i in range(len(x)):
    y = predict(network,x)
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt+=1
print("Accuracy "+str(float(accuracy_cnt)/len(x)))
# mini-batch
batch_size = 100
accuracy_cnt = 0
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt +=np.sum(p==t[i:i+batch_size])

print("Accuracy "+str(float(accuracy_cnt)/len(x)))
