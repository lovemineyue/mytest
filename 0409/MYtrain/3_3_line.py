from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

from mxnet.gluon import data as gdata

batch = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch, shuffle=True)


# for data in dataload:
#     # print(data.shape)
#     x , y  = data
#     print(x.shape, y.shape)
#     break

from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))

from mxnet import init
net.initialize(init.Normal(0.01))

from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()

from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})

num_epochs = 3

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(batch)
    l= loss(net(features),labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))


dense = net[0]
print(dense.weight.data(), dense.bias.data())
