#!/usr/bin/env python
#!coding=utf-8
#!@Author : buyao

from mxnet import nd
x = nd.arange(12)
# print(x)


print(x.shape)
print(x.size)

X = x.reshape((3,4))
print(X)

nd.zeros((2,3,4))
nd.ones((3,4))

Y = nd.array([[2,1,4,3],[1,2,3,4],[4,3,2,1]])

print(Y)

nd.random.normal(0,1,shape=(3,4))


X+ Y

X.exp()

nd.dot(X, Y.T)
nd.concat(X, Y,dim=0);nd.concat(X,Y,dim=1)

X == Y

X.sum()
# L2 范数结果
X.norm().asscalar()


A = nd.arange(3),reshape((3,1))
B = nd.arange(2).reshape((1,2))

A+B

X[1:3]


X[1,2] = 9

X[1:2,:]=12



before = id(Y)
Y = Y+X
id(Y) == before
# False

Z= Y.zeros_like()
before = id(Z)
Z[:] = X + Y
id(Z) == before
# True

nd.elemwise_add(X, Y,out=Z)
id(Z) == before
# True

before = id(X)
X +=Y
id(X) == before
# True

import numpy as np

P = np.ones((2,3))
D = nd.array(P)

D.asnumpy()
