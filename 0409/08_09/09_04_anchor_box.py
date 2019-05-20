import d2lzh as d2l
from mxnet import contrib, gluon, image, nd
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(2)

img = image.imread('../../img/catdog.jpg').asnumpy()
h, w = img.shape[0:2]
# print(img.shape)   (561, 728, 3)

# print(h, w)
# 构建一个批次的的输入数据
X = nd.random.uniform(shape=(1, 3, h, w))  # 构造输入数据

# 在某幅图中中 产生anchor box
Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape)

# 5 = 5个 anchor box   ，4等于 左上和右下坐标 ，坐标都除以了对应的宽高
boxes = Y.reshape((h, w, 5, 4))
# print(boxes.shape)
bbox = boxes[250, 250, 0, :]
# print(bbox) #[0.06 0.07 0.63 0.82] <NDArray 4 @cpu(0)>


# 描绘图像中以某个像素为中心的所有锚框
# 本函数已保存在d2lzh包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

d2l.set_figsize()
bbox_scale = nd.array((w, h, w, h))
print(bbox_scale)
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
# plt.show()

# 标注训练的anchor box
# 真实的框
ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
# anchor box
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
# plt.show()

#标注 锚框标注类别和偏移量
labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0),
                                   ground_truth.expand_dims(axis=0),
                                   nd.zeros((1, 3, 5)))
# 锚框标注的类别, 0自然背景, 1狗, 2猫
print(labels[2])

# mask 掩码（mask）变量, 形状为(批量大小, 锚框个数的四倍)
labels[1]

# 每个锚框标注的四个偏移量，其中负类锚框的偏移量标注为0
labels[0]

# 输出预测边界框
# 构造anchor box
anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
# 构造ofset 假设为 0
offset_preds = nd.array([0] * anchors.size)

cls_probs = nd.array([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

# 非极大值抑制并设阈值为0.5
output = contrib.ndarray.MultiBoxDetection(
    cls_probs.expand_dims(axis=0), offset_preds.expand_dims(axis=0),
    anchors.expand_dims(axis=0), nms_threshold=0.5)

# 第一个元素是索引从0开始计数的预测类别（0为狗，1为猫），
# 其中-1表示背景或在非极大值抑制中被移除。第二个元素是预测边界框的置信度
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].asnumpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)
