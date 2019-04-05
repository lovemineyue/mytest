from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


# layer = CenteredLayer()
# a= layer(nd.array([1, 2, 3, 4, 5]))
# print(a)


# net = nn.Sequential()
# net.add(nn.Dense(128),
#         CenteredLayer())
#
# net.initialize()
# y = net(nd.random.uniform(shape=(4, 8)))
# b = y.mean().asscalar()
#
# print(b)



# 4.4.2. 含模型参数的自定义层
# params = gluon.ParameterDict()
# params.get('param2', shape=(2, 3))
# print(params)


class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


dense = MyDense(units=3, in_units=5)
# print(dense.params)


dense.initialize()
c = dense(nd.random.uniform(shape=(2, 5)))
# print(c)


net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
d = net(nd.random.uniform(shape=(2, 64)))

print(d)
