from mxnet.gluon import nn, data as gdata ,loss as gloss
import mxnet as mx
from mxnet import init, autograd, nd, gluon
import d2lzh as d2l

net = nn.Sequential()

net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))

batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
#
for i in range(num_epochs):
    train_l , train_acce, n = 0.0, 0.0 ,0
    for X, y in test_iter:
        with autograd.record():
            l = loss(net(X), y).sum()
            l.backward()
        trainer.step(batch_size)
        y = y.astype("float32")

        train_l += l.asscalar()
        train_acce += (net(X).argmax(axis=1)==y).sum().asscalar()
        n+= y.size

    test_acce, test_n = 0, 0
    for test_x, test_y in test_iter:
        test_y = test_y.astype("float32")
        test_acce += (net(test_x).argmax(axis=1)==test_y).sum().asscalar()
        test_n += test_y.size
    print("epoch ={}, loss{:.4}, acc{:.2}, test_acc{:.2}".format(num_epochs, train_l/n, train_acce/n, test_acce/test_n))


# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
#               None, trainer)
