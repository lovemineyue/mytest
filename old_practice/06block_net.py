from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

X =  nd.random.uniform(shape=(2,20))
# net = MLP()
# net.initialize()
# print(net(X))


class Mysequential(nn.Block):
    def __init__(self, **kwargs):
        super(Mysequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x

# net = Mysequential()
#
# net.add(nn.Dense(256, activation='relu'))
# net.add(nn.Dense(10))
# net.initialize()
# net(X)
#
# print(net(X))

class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
        'rand_weight',nd.random.uniform(shape=(10,20)))
        self.dense = nn.Dense(10, activation='relu')

        # self.dense = nn.Dense(10, activation='relu')
    def forward(self, x):
        x = self.dense(x)
        # print(x.shape) #(2, 10)
        x = nd.relu(nd.dot(x, self.rand_weight.data())+1)
        x = self.dense(x)

        while x.norm().asscalar() > 1:
            x /=2
        if x.norm().asscalar() < 0.8:
            x*=10
        return x.sum()
#
# net = FancyMLP()
# net.initialize()
# print(net(X))


class NestNLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestNLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        print('---initialize---')
        self.net.add(nn.Dense(64, activation='relu'),
                    nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

    # def add(self, block):
    #     print("=====add=====")
    #     self._children[block.name] = block
    #
    # def forward(self, x):
    #     print("=====forward====")
    #     for block in self._children.values():
    #         x = block(x)
    #
    #     print(x.shape)
    #     return self.dense(self.net(x))


net = nn.Sequential()
net.add(NestNLP(), nn.Dense(20), FancyMLP())
net.initialize()

print (net(X))


# net = NestNLP()
# net.add(nn.Dense(20))
# net.initialize()
#
# print(net(X))
