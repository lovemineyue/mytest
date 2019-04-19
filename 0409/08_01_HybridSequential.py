from mxnet import nd, sym
from mxnet.gluon import nn
import time

def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))

    net.initialize()
    return net

x = nd.random.uniform(shape=(1,512))
# net = get_net()
# net(x)
# net.hybridize()
# net(x)

def benchmark(net, x):
    start = time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall()  # 等待所有计算完成方便计时
    return time.time() - start

# net = get_net()
# print('before hybridizing: %.4f sec' % (benchmark(net, x)))
# net.hybridize()
# print('after hybridizing: %.4f sec' % (benchmark(net, x)))
#
#
# net.export('my_mlp')
#
# x = sym.var('data')
# # print(net(x))

class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # x.asnumpy()
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)


net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
# net(x)
#
# net(x)

net.hybridize()
net(x)

net(x)
