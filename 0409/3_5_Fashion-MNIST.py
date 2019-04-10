import sys
import d2lzh as d2l
from mxnet.gluon import data as gdata
import time
from matplotlib import pyplot as plt


mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test  = gdata.vision.FashionMNIST(train=False)

test = len(mnist_test)
train = len(mnist_train)
print(test, train)

feature, label = mnist_train[0]

print(feature.shape , feature.dtype)

print(label, type(label),label.dtype)

# plt.imshow(feature)
# plt.show()

# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

X , y = mnist_train[:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
# plt.show()

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle = True)
test_iter  = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle = True)

start = time.time()
for X, y in train_iter:
    continue

print ('%.2f sec'%(time.time() - start))
