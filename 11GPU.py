import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu(), mx.gpu())

x = nd.array([1, 2, 3])

# print(x.context)

# a = nd.array([1, 2, 3], ctx=mx.gpu())

# print(a)


# B = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))

# 除了在创建时指定，我们也可以通过copyto函数和as_in_context函数在设备之间传输数据。下面我们将内存上的NDArray变量x复制到gpu(0)上。
# y = x.copyto(mx.gpu())

# as_in_context函数在设备之间传输数据
# z = x.as_in_context(mx.gpu())

# 需要区分的是，如果源变量和目标变量的context一致，as_in_context函数使目标变量和源变量共享源变量的内存或显存。
# y.as_in_context(mx.gpu()) is y

# 而copyto函数总是为目标变量开新的内存或显存。
# y.copyto(mx.gpu()) is y


# MXNet的计算会在数据的context属性所指定的设备上执行。为了使用GPU计算，我们只需要事先将数据存储在显存上。计算结果会自动保存在同一块显卡的显存上。
# (z + 2).exp() * y

#注意，MXNet要求计算的所有输入数据都在内存或同一块显卡的显存上。这样设计的原因是CPU和不同的GPU之间的数据交互通常比较耗时。因此，MXNet希望用户
#确切地指明计算的输入数据都在内存或同一块显卡的显存上。例如，如果将内存上的NDArray变量x和显存上的NDArray变量y做运算，会出现错误信息。当我们打印
#NDArray或将NDArray转换成NumPy格式时，如果数据不在内存里，MXNet会将它先复制到内存，从而造成额外的传输开销。


# Gluon的GPU计算¶
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())

# 当输入是显存上的NDArray时，Gluon会在同一块显卡的显存上计算结果。
net(y)

# 下面我们确认一下模型参数存储在同一块显卡的显存上。
net[0].weight.data()
