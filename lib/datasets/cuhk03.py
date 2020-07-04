# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import glob
import re
import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
from utils.utils import mkdir


class CUHK03(BaseDataset):
    def __init__(self,
                 root,
                 list_path=None,
                 num_samples=None,
                 num_classes=None,
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

        super(CUHK03, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.class_weights = None
        self.multi_scale = multi_scale
        self.flip = flip

        self.train_dir = os.path.join(self.root, 'cuhk03', 'detected', 'bounding_box_train')
        self.query_dir = os.path.join(self.root, 'cuhk03', 'detected', 'query')
        self.gallery_dir = os.path.join(self.root, 'cuhk03', 'detected', 'bounding_box_test')

        # self.train_dir = os.path.join(self.root, 'cuhk03', 'labeled', 'bounding_box_train')
        # self.query_dir = os.path.join(self.root, 'cuhk03', 'labeled', 'query')
        # self.gallery_dir = os.path.join(self.root, 'cuhk03', 'labeled', 'bounding_box_test')

        self.files = self._process_dir(self.train_dir)
        # self.query = self._process_dir(self.query_dir)
        # self.gallery = self._process_dir(self.gallery_dir)

    def _process_dir(self, dir_path):
        img_paths = glob.glob(os.path.join(dir_path, '*.png'))

        dataset = []
        for img_path in img_paths:
            dataset.append(img_path)

        return dataset

    def resize_image(self, image, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return image

    def gen_sample_without_label(self, image, multi_scale=True,
                                 is_flip=True, center_crop_test=False):
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]

        return image

    def __getitem__(self, index):
        img_path = self.files[index]
        name = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        size = image.shape[:-1]

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]

        image_size = (self.crop_size[1], self.crop_size[0])
        image = self.resize_image(image, image_size)
        image = self.gen_sample_without_label(image, self.multi_scale, False)

        return image.copy(), image.copy(), np.array(size), name

    def multi_scale_inference(self, model, image, scales=False, flip=False):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred,
                          size=(size[-2], size[-1]),
                          mode='bilinear')
        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output,
                                     size=(size[-2], size[-1]),
                                     mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
            flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
            flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
            flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
            flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
            flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def largest_connect_component(self, img):
        area_thr = 100
        import skimage.measure as M
        mask = np.zeros_like(img)
        mask[img > 0] = 1
        labeled_mask, num = M.label(mask, neighbors=8, background=0, return_num=True)
        if num <= 1:
            return img
        else:
            for i in range(1, num + 1):
                if np.sum(labeled_mask == i) < area_thr:
                    mask[labeled_mask == i] = 0
            img = img * mask
            return img

    def combine_label(self, img):
        img[img == 6] = 1
        img[img == 7] = 1
        img[img == 8] = 0
        img[img == 9] = 0
        return img

    def remove_small_area(self, img):
        area_thr = 80
        import skimage.measure as M
        label_set = set(img.flatten())
        for label in label_set:
            if label == 0:
                continue
            mask = np.zeros_like(img)
            mask[img == label] = 1
            labeled_mask, num = M.label(mask, neighbors=8, background=0, return_num=True)

            for i in range(1, num + 1):
                if np.sum(labeled_mask == i) >= area_thr:
                    mask[labeled_mask == i] = 0
            img[mask > 0] = 0
        return img

    def remove_duplicate_area(self, img):
        num_thr = 2
        import skimage.measure as M
        label_set = set(img.flatten())
        for label in label_set:
            if label == 0:
                continue
            mask = np.zeros_like(img)
            mask[img == label] = 1
            labeled_mask, num = M.label(mask, neighbors=8, background=0, return_num=True)

            if num <= num_thr:
                continue

            area = [-np.sum(labeled_mask == i) for i in range(1, num + 1)]
            match = np.argsort(area) + 1
            for i in match[:num_thr]:
                mask[labeled_mask == i] = 0
            img[mask > 0] = 0
        return img

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]
            pred = self.largest_connect_component(pred)
            pred = self.combine_label(pred)
            pred = self.remove_small_area(pred)
            pred = self.remove_duplicate_area(pred)

            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            dataset_name, file_name = name[i].split('/')
            path = os.path.join(sv_path, dataset_name)
            mkdir(path)
            save_img.save(os.path.join(path, file_name.replace('.png', '.jpg')))
