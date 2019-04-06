import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd, gluon, image, init
from mxnet.gluon import data as gdata , loss as gloss , utils as gutils
import sys
import time
from matplotlib import pyplot as plt

d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
# d2l.plt.imshow(img.asnumpy())


# 本函数已保存在d2lzh包中方便以后使用
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

# apply(img, gdata.vision.transforms.RandomFlipLeftRight())
# apply(img, gdata.vision.transforms.RandomFlipTopBottom())


shape_aug = gdata.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# apply(img, gdata.vision.transforms.RandomBrightness(0.5))
# apply(img, gdata.vision.transforms.RandomHue(0.5))

# 我们也可以创建RandomColorJitter实例并同时设置如何随机变化图像的
# 亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。

color_aug = gdata.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)


augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
# apply(img, augs)
# plt.show()

show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=0.8)
