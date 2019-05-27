import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from mxnet import autograd

batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


net  = nn.Sequential()
net.add(nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

Trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": 0.01})


num_epochs = 3

for i in range(1, num_epochs+1):
    train_l, train_acc, assc, n = 0.0, 0.0, 0.0, 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y).sum()
        l.backward()
        Trainer.step(batch_size)
        y = y.astype("float32")
        train_l += l.asscalar()
        train_acc += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size

    test_n = 0
    for test_X, test_y in test_iter:
        test_y = test_y.astype("float32")
        assc += (net(test_X).argmax(axis=1) == test_y).sum().asscalar()
        test_n += test_y.size

    assc/=test_n
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (i , train_l / n, train_acc / n, assc))


# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
#               None, trainer)
