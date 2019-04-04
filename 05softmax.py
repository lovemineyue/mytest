# from mxnet import nd
import matplotlib.pyplot as plt
# import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import loss, nn


# x = nd.random.normal(scale=1,shape=(10000,3,4))
# y = nd.random.normal(scale=1,shape=(10000,3,4))
# plt.figure()
# plt.scatter(x.asnumpy(),y.asnumpy(),1)
# plt.show()

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'),
        nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))
