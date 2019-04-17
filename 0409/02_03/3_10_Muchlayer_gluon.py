from mxnet import nd, autograd, init ,gluon
from mxnet.gluon import loss as gloss, data as gdata, nn
import d2lzh as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

net = nn.Sequential()
# net.add(nn.Dense(num_hiddens,activation='relu'))
# net.add(nn.Dense(num_outputs))
net.add(nn.Dense(64, activation='sigmoid'),
        nn.Dense(32,activation='sigmoid'),
        nn.Dense(10))

# net.initialize(init.Normal(sigma=0.01))
net.initialize(init.Xavier())

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':0.02})

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
