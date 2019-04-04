#!/usr/bin/env python
#!coding=utf-8
#!@Author : buyao

# from mxnet import autograd, nd
import d2lzh as d2l
from mxnet.gluon import loss as gloss, nn

y_hat = nd.array([[0.1, 0.3, 0.6],[0.3, 0.2, 0.5]])
y = nd.array([0,2],dtype='int32')

result = - nd.pick(y_hat,y).log()

print(result)

s = nd.exp(nd.array([2])).log()
print(s)