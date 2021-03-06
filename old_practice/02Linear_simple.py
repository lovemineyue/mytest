from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from mxnet.gluon import data as gdata
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2

features = nd.random.normal(scale=1,shape=(num_examples, num_inputs))
labels = true_w[0] * features[:,0] + true_w[1]*features[:,1] + true_b
labels += nd.random.normal(scale=0.01,shape=(labels.shape))


batch_size = 10

dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
# for i,(X, y) in enumerate(data_iter):
#     print(X, y)
#     if i ==2:
#         break

from mxnet.gluon import nn
from mxnet import init

net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01))
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()


from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})

num_epochs = 3

for epoch in range(1,num_epochs+1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f'%(epoch, l.mean().asnumpy()))


dense=net[0]
print(dense.weight.data())

print(dense.bias.data())


