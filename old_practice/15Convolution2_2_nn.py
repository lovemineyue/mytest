from mxnet import nd , autograd
from mxnet.gluon import nn

def corr2d(X, K):
    h , w = K.shape

    Y = nd.zeros((X.shape[0]-h + 1, X.shape[1]- w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j ] = (X[i:i+h, j:j+w]*K).sum()

    return Y

# X = nd.array([[0,1,2],[3,4,5],[6,7,8]])
K = nd.array([[0,1],[2,3]])
#
# result = corr2d(X,K)
# print(result, X.shape)
#

class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=kernel_size)
        self.bias = self.params.get('bias',shape=(1,))

    def forward(self, x):
        #return corr2d(x, self.weight.data()) + self.bias.data()

        data = x.reshape((1,1) + x.shape)
        # print(data.shape)
        weight = self.weight.data()
        print(weight)
        weight = weight.reshape((1,1) + weight.shape)
        bias = self.bias.data()
        kernel = self.weight.shape

        return nd.Convolution(data, weight, bias, kernel, stride=(1,1),dilate=(1,1),pad=(0,0),num_filter=1)


X = nd.ones((6, 8))
X[:, 2:6] = 0

K = nd.array([[1,-1]])

Y = corr2d(X ,K)

# print(Y.shape)
# conv2d = nn.Conv2D(1, kernel_size=(1,2))
# conv2d.initialize()

X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))



# for i in range(10):
#     with autograd.record():
#         y_hat = conv2d(X)
#         l = (y_hat - Y)**2
#     l.backward()
#
#     conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
#     if (i+1)%2 == 0:
#         print('batch %d, loss %.3f'%(i+1, l.sum().asscalar()))
#
# a = conv2d.weight.data().reshape((1,2))
# print(a)


# 作业
X = X.reshape((6,8))
Y = Y.reshape((6,7))
conv2d = Conv2D(kernel_size=(1,2))
conv2d.initialize()
# Y_hat = conv2d(X)

# Y_hat.backward()
for i in range(10):
    with autograd.record():
        y_hat = conv2d(X)
        l = (y_hat - Y)**2
    l.backward()
