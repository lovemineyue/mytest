# class VOCDetection():
#
#     CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
#     def __init__(self):
#         pass
#     @property
#     def classes(self):
#         return type(self).CLASSES
#     def __delattr__(self):
#         pass
#
# print(VOCDetection().classes)


# -*- coding: UTF-8 -*-
class Entity(object):
    """
    调用实体来改变实体的位置
    """
def __init__(self, size, x, y):
    self.x, self.y = x, y
    self.size = size

def __call__(self, x, y):
    """
    改变实体的位置
    """
    self.x, self.y = x, y


enti = Entity(size = 3, x = 4, y = 5)

enti(3,4)
