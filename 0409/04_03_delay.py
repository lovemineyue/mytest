from mxnet import nd, init
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print("init", name, data.shape)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(MyInit())

# X = nd.random.uniform(shape=(2,20))
# Y = net(X)

net.initialize(init=MyInit(), force_reinit = True)

# net = nn.Sequential()
# net.add(nn.Dense(256, in_units=20, activation='relu'),
#         nn.Dense(10,in_units=256))
#
# net.initialize(MyInit())
