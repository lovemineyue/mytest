import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.hidden = nn.Dense(256,activation="relu")
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


# net = MLP()
# net.initialize()
X = nd.random.uniform(shape=(2,20))
# print(net(X))


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)

        return x

#
# net = MySequential()
# net.add(nn.Dense(256, activation="relu"))
# net.add(nn.Dense(10))
# net.initialize()
#
# print(net(X))



class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        print(x.shape)
        # 使用创建的常数参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，这里我们需要调用asscalar函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


net = FancyMLP()
# net.initialize()
# print(net(X))


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation="relu"),
                    nn.Dense(32,activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):

        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(),nn.Dense(20), FancyMLP())

net.initialize()
print(net(X))
