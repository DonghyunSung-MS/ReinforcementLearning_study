import numpy as np
from common.net_class import TwoLayerNet
from dataset.mnist import load_mnist
print()
net = TwoLayerNet(input_dim = 784, hidden_dim = 100, output_dim = 10)
x = np.random.rand(10,784)
t = np.random.rand(10,10)
y= net.predictByLayer(x)
print()
print("Test : prediction shape (batch_size,output_dim)")
print(y.shape)
#print(net.numerical_gradient(x,t))
print()

##---------------------------------train algorithms numerical_gradient------------------
if 0:
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
    train_loss_history = []

    #hyper parameters
    iters_num = 5 ## due to computing speed in my desktop
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    network = TwoLayerNet(input_dim = 784, hidden_dim = 50, output_dim = 10)

    for _ in range(iters_num):
        batch_mask = np.random.choice(train_size,batch_size) # shuffle index
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]













wesdafsafsdfs






















        grad = network.numerical_gradient(x_batch, t_batch)

        for key in ('W1','b1','W2','b2'):
            #print(key)
            network.params[key]-=learning_rate*grad[key]

        loss = network.loss(x_batch,t_batch)
        train_loss_history.append(loss)

    import matplotlib.pyplot as plt

    plt.plot(train_loss_history)
    plt.show()

##---------------------------------train algorithms gradient------------------
if 1:
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
    train_loss_history = []
    train_acc_history = []
    test_acc_history = []

    #hyper parameters
    iters_num = 10000 ## due to computing speed in my desktop
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    iter_per_epoch = max(train_size/batch_size,1)

    network = TwoLayerNet(input_dim = 784, hidden_dim = 50, output_dim = 10)
    #print(network.accuracy(x_train,t_train))

    for _ in range(iters_num):
        batch_mask = np.random.choice(train_size,batch_size) # shuffle index
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        #print(x_batch)
        grad = network.gradient(x_batch, t_batch)

        for key in ('W1','b1','W2','b2'):
            #print(key)
            network.params[key]-=learning_rate*grad[key]
            #print("gradient : "+ str(grad[key]))
        loss = network.loss(x_batch,t_batch)
        #print("loss : " + str(loss))
        train_loss_history.append(loss)
        #print()
        if _%iter_per_epoch == 0:
            train_acc = network.accuracy(x_train,t_train)
            test_acc = network.accuracy(x_test,t_test)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(test_acc_history, label="test")
    ax1.plot(train_acc_history, label="train")
    ax2.plot(train_loss_history)
    ax1.set_title("Train Accuracy")
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend(loc='best')
    ax2.set_title("Train loss")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("iteration")
    plt.show()
