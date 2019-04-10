import d2lzh as d2l
from mxnet import nd, gluon, autograd,init
from mxnet.gluon import data as gdata , loss as gloss ,nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

num_epochs = 3

def evaluate_accuracy(data_iter ,net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1)==y).sum().asscalar()
        n += y.size
    return acc_sum / n

for epoch in range(1, num_epochs+1):
    train_l_sum, train_acc_sum, n = 0.0, 0.0 , 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y).sum()
        l.backward()
        trainer.step(batch_size)
        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
        n += y.size
    test_acc = evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %f, train acc%.3f, test acc %.3f'%(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))

# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
#               None, trainer)
