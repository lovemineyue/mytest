#!/usr/bin/env python
#!coding=utf-8
#!@Author : buyao

from mxnet import nd ,autograd
from time import time
from matplotlib import  pyplot as plt

# a = nd.ones(shape=1000)
# b = nd.ones(shape=1000)
#
# start = time()
# c = nd.zeros(shape=1000)
# for i in range(1000):
#     c[i] = a[i] + b[i]
#
# print(time() - start)

# test 2
start = time()
a = nd.ones(shape=1000)
b = 100
c = a+b
print(time() - start)


# 作业
import numpy as np
start1 = time()
np.ones(shape=1000)
b = 100
c = a + b
print(time() - start1)