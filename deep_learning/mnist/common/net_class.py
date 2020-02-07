import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import * # numerical_gradient
from common.layers import * #ReluLayer, SigmoidLayer, AffineLayer, SoftMaxWithLoss etc
from collections import OrderedDict

class SimpleNet:

    def __init__(self,label, input_dim = 2,ouput_dim = 3):
        print("Simple network fully connect -> softmax")
        self.W = np.random.randn(input_dim, ouput_dim)
        self.label = label
    def predict(self,x):
        z = np.dot(x,self.W)
        return z

    def loss(self,x,t):
        z = self.predict(x) # if u twist the knob W(weights), it affect the loss in here.
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss

class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim, weight_init_std = 0.01):
        print("Two layer network fully connect -> activation(relu)-> fully connect -> softmax")
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_dim, output_dim)
        self.params['b2'] = np.zeros(output_dim)

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = ReluLayer()
        self.layers['Affine2'] =  AffineLayer(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftMaxWithLoss(loss='CE')

    def predictByLayer(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def predictAtOnce(self,x):
        print("It is not recommanded method")
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    def loss(self, x, t):
        #y = self.predictAtOnce(x)
        #loss = cross_entropy_error(y,t)
        y = self.predictByLayer(x)
        loss = self.lastLayer.forward(y,t)
        return loss

    def accuracy(self, x, t):
        y = self.predictByLayer(x) # probability
        y = np.argmax(y,axis = 1) # index
        if t.ndim !=1:
            t = np.argmax(t,axis = 1) # one-hot -> index

        accuracy = np.sum(y==t)/float(x.shape[0]) #x.shape[0] -> batch_size
        return accuracy

    def gradient(self, x, t):
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads={}
        grads['W1'] = self.layers['Affine1'].rLrW
        grads['b1'] = self.layers['Affine1'].rLrb
        grads['W2'] = self.layers['Affine2'].rLrW
        grads['b2'] = self.layers['Affine2'].rLrb

        return grads

    def numerical_gradient(self, x, t):
        print("It is not recommanded method")
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
