#!/usr/bin/env python
#!coding=utf-8
#!@Author : buyao

# from mxnet import autograd, nd
import d2lzh as d2l
from mxnet.gluon import loss as gloss, nn
from mxnet import nd

y_hat = nd.array([[0.1, 0.3, 0.6],[0.3, 0.2, 0.5]])
y = nd.array([0,2],dtype='int32')

result = - nd.pick(y_hat,y).log()

print(result)

s = nd.exp(nd.array([2])).log()
print(s)



import matplotlib.pyplot as plt
# import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

# x = nd.random.normal(scale=1,shape=(10000,3,4))
# y = nd.random.normal(scale=1,shape=(10000,3,4))
# plt.figure()
# plt.scatter(x.asnumpy(),y.asnumpy(),1)
# plt.show()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'),
        nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
