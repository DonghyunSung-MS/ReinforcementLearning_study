import numpy as np
from common.net_class import SimpleNet # class
from common.functions import *

input = np.array([0.6,0.9,0.2]) # ex. normalize(0,1) input
label = np.array([0, 1, 0, 0])
input_dim = input.shape[0]
output_dim = 4
print()
net = SimpleNet(label,input_dim, output_dim)
print("weights")
print(net.W) # random normal init
print()
z = net.predict(input)
print("prediction")
print(z)
print()
y = softmax(z)
print("output, normalize with softmax")
print(y)
print(np.sum(y))
print()

loss = net.loss(input,label) #cross_entropy_error
print("loss function")
print(loss)

f = lambda w: net.loss(input,label)
dW = numerical_gradient(f,net.W)

print(dW[0][0])
