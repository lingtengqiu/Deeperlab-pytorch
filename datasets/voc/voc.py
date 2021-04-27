#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/12/16 下午8:41
# @Author  : 
# @Contact : 
# @File    : voc.py

from datasets.BaseDataset import BaseDataset


class VOC(BaseDataset):
    @classmethod
    def get_class_colors(*args):
        return [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128],
                [0, 0, 64], [0, 0, 192], [0, 128, 64],
                [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0],
                [0, 192, 128], [128, 64, 0], ]

    @classmethod
    def get_class_names(*args):
        return ['background', 'aeroplane', 'bicycle', 'bird',
                'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable',
                'dog', 'horse', 'motorbike', 'person',
                'pottedplant',
                'sheep', 'sofa', 'train', 'tv/monitor']


if __name__ == "__main__":
    data_setting = {'img_root': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG/',
                    'gt_root': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG',
                    'train_source': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG/config/train.txt',
                    'eval_source': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG/config/val.txt'}
    voc = VOC(data_setting, 'train', None)
    print(voc.get_class_names())
    print(voc.get_length())
    print(next(iter(voc)))
