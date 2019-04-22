from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, nn
import os
import subprocess
import time

# a = nd.ones((1, 2))
# b = nd.ones((1, 2))
# c = a * b + 2
# print(c)

class Benchmark():  # 本类已保存在d2lzh包中方便以后使用
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''
        # print("__init__")

    def __enter__(self):
        self.start = time.time()
        # print("__enter__")

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))



# with Benchmark('Workloads are queued.'):
    # x = nd.random.uniform(shape=(2000, 2000))
    # y = nd.dot(x, x).sum()

# with Benchmark('Workloads are finished.'):
#     print('sum =', y)


# with Benchmark():
#     x = nd.random.uniform(shape=(2000, 2000))
#     y = nd.dot(x, x)
#     y.wait_to_read()

# with Benchmark():
    # x = nd.random.uniform(shape=(2000, 2000))
#     y = nd.dot(x, x)
#     z = nd.dot(x, x)
#     nd.waitall()

# with Benchmark():
#     y = nd.dot(x, x)
#     y.asnumpy()
#
#
# with Benchmark():
#     y = nd.dot(x, x)
#     y.norm().asscalar()

# 上面介绍的wait_to_read函数、waitall函数、asnumpy函数、asscalar函数和print函数会触发让前端等待后端计算结果的行为。
# 这类函数通常称为同步函数


x = nd.random.uniform(shape=(2000, 2000))

with Benchmark('synchronous.'):
    for _ in range(2000):
        y = x + 1
        y.wait_to_read()

with Benchmark('asynchronous.'):
    for _ in range(2000):
        y = x + 1
    y.wait_to_read()
    # nd.waitall()


# def data_iter():
#     start = time.time()
#     num_batches, batch_size = 100, 1024
#     for i in range(num_batches):
#         X = nd.random.normal(shape=(batch_size, 512))
#         y = nd.ones((batch_size,))
#         yield X, y
#         if (i + 1) % 50 == 0:
#             print('batch %d, time %f sec' % (i + 1, time.time() - start))
#
#
# net = nn.Sequential()
# net.add(nn.Dense(2048, activation='relu'),
#         nn.Dense(512, activation='relu'),
#         nn.Dense(1))
# net.initialize()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})
# loss = gloss.L2Loss()
#
# def get_mem():
#     res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
#     return int(str(res).split()[15]) / 1e3
#
# for X, y in data_iter():
#     break
# loss(y, net(X)).wait_to_read()
#
# l_sum, mem = 0, get_mem()
# for X, y in data_iter():
#     with autograd.record():
#         l = loss(y, net(X))
#     l_sum += l.mean().asscalar()  # 使用同步函数asscalar
#     l.backward()
#     trainer.step(X.shape[0])
# nd.waitall()
# print('increased memory: %f MB' % (get_mem() - mem))
#
#
# mem = get_mem()
# for X, y in data_iter():
#     with autograd.record():
#         l = loss(y, net(X))
#     l.backward()
#     trainer.step(X.shape[0])
# nd.waitall()
# print('increased memory: %f MB' % (get_mem() - mem))
