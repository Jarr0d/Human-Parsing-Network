# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import random
import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class MHP(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(MHP, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            if 'train' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name,
                          "split": 'train'}
            elif 'val' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name,
                          "split": 'val'}
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 1.0 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image,
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label,
                               None,
                               fx=self.downsample_rate,
                               fy=self.downsample_rate,
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(
            self.root, 'LV-MHP-v2', item["split"], item["img"]),
            cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(
            self.root, 'LV-MHP-v2', item["split"], item["label"]),
            cv2.IMREAD_GRAYSCALE)
        size = label.shape

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

        image_size = (self.crop_size[1], self.crop_size[0])
        image, label = self.resize_image(image, label, image_size)
        image, label = self.gen_sample(image, label,
                                       self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred,
                          size=(size[-2], size[-1]),
                          mode='bilinear')
        assert flip == False
        return pred.exp()
