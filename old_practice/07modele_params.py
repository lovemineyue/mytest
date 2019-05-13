from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

X = nd.random.uniform(shape=(2, 20))
Y = net(X)  # 前向计算

print(net[0].params, type(net[0].params))


# print(net[0].params['dense0_weight'], net[0].weight)

# print(net[0].weight.data())

# print(net[0].weight.grad())

# print(net[1].bias.data())


# print(net.collect_params())

# 这个函数可以通过正则表达式来匹配参数名，从而筛选需要的参数
# print(net.collect_params('.*weight'))


# 4.2.2. 初始化模型参数
# 我们将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。
# net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)

# print(net[0].weight.data()[0])



# net.initialize(init=init.Constant(1), force_reinit=True)
# print(net[0].weight.data()[0])
#
# net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
# print(net[0].weight.data()[0])


# 4.2.3. 自定义初始化方法
# class MyInit(init.Initializer):
#     def _init_weight(self, name, data):
#         # print('Init', name, data.shape)
#         data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
#         data *= data.abs() >= 5
# #
# net.initialize(MyInit(), force_reinit=True)
# # print(net[0].weight.data()[0])
#
#
# net[0].weight.set_data(net[0].weight.data() + 10)
# b = net[0].weight.data()[0]
# print(b)

# 4.2.4. 共享模型参数
# net = nn.Sequential()
# shared = nn.Dense(8, activation='relu')
# net.add(nn.Dense(8, activation='relu'),
#         shared,
#         nn.Dense(8, activation='relu', params=shared.params),
#         nn.Dense(10))
# net.initialize()
#
# X = nd.random.uniform(shape=(2, 20))
# net(X)
#
# print(net[1].weight.data()[0] == net[2].weight.data()[0])
