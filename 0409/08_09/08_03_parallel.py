import d2lzh as d2l
import mxnet as mx
from mxnet import nd

def run(x):
    return [nd.dot(x, x) for _ in range(10)]

x_cpu = nd.random.uniform(shape=(2000,2000))
x_gpu = nd.random.uniform(shape=(2000,2000),ctx=mx.gpu(0))

# run(x_cpu)
# run(x_gpu)
# nd.waitall()
#
# with d2l.Benchmark('Run on CPU'):
#     run(x_cpu)
#     nd.waitall()
#
# with d2l.Benchmark("Then run on GPU"):
#     run(x_gpu)
#     nd.waitall()
#
#
# with d2l.Benchmark("Then run on GPU and CPU"):
#     run(x_gpu)
#     run(x_cpu)
#     nd.waitall()


def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]

# with d2l.Benchmark('Run on gpu'):
#     y = run(x_gpu)
#     nd.waitall()
#
# with d2l.Benchmark("then copy to cpu"):
#     copy_to_cpu(y)
#     nd.waitall()

with d2l.Benchmark("then copy to cpu"):
    y=run(x_gpu)
    copy_to_cpu(y)
    nd.waitall()
