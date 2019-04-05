# 4.3.1. 延后初始化

from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了
#
# net = nn.Sequential()
# net.add(nn.Dense(256, activation='relu'),
#         nn.Dense(10))
#
# net.initialize(init=MyInit())
# X = nd.random.uniform(shape=(2, 20))
# Y = net(X)

# 避免延后初始化¶
net.initialize(init=MyInit(), force_reinit=True)


# net = nn.Sequential()
# net.add(nn.Dense(256, in_units=20, activation='relu'))
# net.add(nn.Dense(10, in_units=256))
#
# net.initialize(init=MyInit())
