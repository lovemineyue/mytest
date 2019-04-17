import mxnet as mx
from mxnet import nd, init
from mxnet.gluon import nn

# print(mx.cpu(), mx.gpu(), mx.gpu(1))

x = nd.array([1,2,3])

# print(x)

# print(x.context)

a = nd.array([1,2,3],ctx=mx.gpu())

# print(a)

B = nd.random.uniform(shape=(2,3), ctx=mx.gpu())
# print(B)

y = x.copyto(mx.gpu())
# print(y)

z = x.as_in_context(mx.gpu())


y1 = y.as_in_context(mx.gpu()) is y
# print(y1)


y2 = y.copyto(mx.gpu()) is y
# print(y2)

yy = (z + 2).exp() * y
# print(yy)


net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())

y3=net(y)

print(y3)

print(net[0].weight.data())
