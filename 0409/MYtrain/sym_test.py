import mxnet as mx


from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)


batch_size = 32
from mxnet.gluon import data as gdata
import random


def data_batch():
    num_examples = len(features)
    indices = list(range(num))
    random.shuffle(indices)
    # print(datalist[0:10])
    for i in range(0, num, batch_size):

        j = nd.array(indices[i:min(i+batch_size,num_examples)])
        yield features.take(j), labels.take(j)


data_batch()
