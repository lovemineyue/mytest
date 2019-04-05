import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu(), mx.gpu())

x = nd.array([1, 2, 3])

print(x.context)

a = nd.array([1, 2, 3], ctx=mx.gpu())

print(a)
