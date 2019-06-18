
import os
import logging
import warnings
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

splits=((2007, 'trainval'),)
root = "/Users/demon/Desktop/learn/learn-py-lib/mxnet/d2l-zh-1.0/mytest/0409/gluoncv_folder/object_detection/VOC/"
anno_path = os.path.join('{}', 'Annotations', '{}.xml')

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

def load_items(splits):
    """Load individual image indices from splits."""
    root = "/Users/demon/Desktop/learn/learn-py-lib/mxnet/d2l-zh-1.0/mytest/0409/gluoncv_folder/object_detection/VOC/"
    ids = []
    for year, name in splits:
        root = os.path.join(root, 'VOC' + str(year))
        lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
        with open(lf, 'r') as f:
            ids += [(root, line.strip()) for line in f.readlines()]
    return ids


# print(load_items(splits)[1])
def readLabel(idx):
    anno_path = os.path.join('{}', 'Annotations', '{}.xml')
    _im_shapes = {}
    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    index_map=None
    index_map = index_map or dict(zip(classes, range(20)))

    img_id = load_items(splits)[idx]
    anno_path = anno_path.format(*img_id)
    root = ET.parse(anno_path).getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    if idx not in _im_shapes:
        # store the shapes for later usage
        _im_shapes[idx] = (width, height)
    label = []
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in classes:
            continue
        cls_id = index_map[cls_name]
        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)
        # print(cls_name,xmin,ymin,xmax,ymax
        print(cls_id)

readLabel(0)
