import d2lzh as d2l
import mxnet as mx
from mxnet import nd

def run(x):
    return [nd.dot(x, x) for _ in range(10)]
