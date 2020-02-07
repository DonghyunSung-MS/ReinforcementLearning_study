import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *

class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        y = x.copy()
        y[self.mask] = 0 # mask 씨워서 True 를 0 으로 만든다.
        return y

    def backward(self, rLry):
        rLry[self.mask] = 0
        rLrx = rLry
        return rLrx

class SigmoidLayer:
    def __init__(self):
        self.y = None

    def forward(self,x):
        self.y = 1.0/(1 + np.exp(-x))
        return self.y

    def backward(self, rLry):
        rLrx = rLry*self.y*(1.0 - self.y)
        return rLrx

class AffineLayer:
    def __init__(self, W, b):
        # y = xw + b   dim: (1, Out) = (1, In) (In, Out) + (1, Out)
        self.W = W #(In, Out)
        self.b = b #(1, Out)
        self.x = None # (1, In)
        self.rLrW = None #(In, Out)
        self.rLrb = None #(1, Out)

    def forward(self, x):
        self.x = x
        return np.dot(x,self.W) + self.b

    def backward(self, rLry):
        self.rLrW = np.dot(self.x.T,rLry) # x' rL/ry
        self.rLrb = np.sum(rLry,axis=0) # <------error
        rLrx = np.dot(rLry, self.W.T)
        return rLrx


class SoftMaxWithLoss:
    def __init__(self,loss='CE'):
        self.lossfunction = loss
        self.loss = None
        self.y = None #prob
        self.t = None #label

    def cross_entropy_error(self,y,t):
        if y.ndim == 1:
            t = t.reshape(1,t.size)
            y = y.reshape(1,y.size)
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = np.argmax(t, axis=1)

        batch_size = y.shape[0]
        # 각 미니 배치 에서 해당 레이블 인덱스만 계산해라 one hot encoding 응용
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def mean_square_error(y,t):
        error = y-t
        output = 0.5*np.sum((y-t)**2)

    def forward(self, x, t):
        #print("error occur? "+str(x.ndim))
        self.y = softmax(x) #softmax
        self.t = t
        if self.lossfunction =='CE':
            self.loss = self.cross_entropy_error(self.y, self.t)

        elif self.lossfunction == 'MSE':
            self.loss = self.mean_square_error(self.y, self.t)

        return self.loss

    def backward(self, dout =1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            rLrx = (self.y - self.t)/batch_size
        else:
            rLrx = self.y.copy()
            rLrx[np.arrange(batch_size), self.t] -=1
            rLrx = rLrx/batch_size
        return rLrx
