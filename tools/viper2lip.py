import os
from collections import defaultdict
import cv2
import numpy as np
import random
from tqdm import tqdm
import skimage.measure as M


"""
0 -> background
1 -> head(hair)
2 -> upper
3 -> lower
4 -> shoes
5 -> bag
6 -> head(hat)
7 -> head(face)
8 -> arm
9 -> leg
"""
convertDict = {43:0, 30:6, 29:5, 95:1, 76:8, 208:2, 179:3, 20:9, 16:4, 188:7,
               48:1}


def find_box(img, delta=2):
    row_val = np.sum(img, axis=0)
    col_val = np.sum(img, axis=1)
    w, h = len(row_val), len(col_val)
    x1, x2, y1, y2 = 0, 0, 0, 0
    for i, pixel in enumerate(row_val):
        if pixel != 0:
            x1 = i
            break
    for i, pixel in enumerate(row_val[::-1]):
        if pixel != 0:
            x2 = w - i
            break
    for i, pixel in enumerate(col_val):
        if pixel != 0:
            y1 = i
            break
    for i, pixel in enumerate(col_val[::-1]):
        if pixel != 0:
            y2 = h - i
            break

    if x1 - delta >= 0:
        x1 = x1 - delta
    if y1 - delta >= 0:
        y1 = y1 - delta
    if x2 + delta <= w:
        x2 = x2 + delta
    if y2 + delta <= h:
        y2 = y2 + delta

    return (x1, y1, x2, y2)


def seperate_hat_bag(label):
    if len(label[label == 29]) == 0:
        return label
    if len(label[label == 188]) > 0:
        mask = np.zeros_like(label)
        mask[label == 188] = 1
        y = find_box(mask, delta=0)[-1]
    elif len(label[label == 208]) > 0:
        mask = np.zeros_like(label)
        mask[label == 208] = 1
        y = find_box(mask, delta=0)[1]

    mask = np.zeros_like(label)
    mask[label == 29] = 1
    labeled_mask, num = M.label(mask, neighbors=8, background=0, return_num=True)
    for i in range(1, num + 1):
        mask = np.zeros_like(labeled_mask)
        mask[labeled_mask == i] = 1
        x1, y1, x2, y2 = find_box(mask, delta=0)
        if (y1 + y2) / 2 <= y:
            label[labeled_mask == i] = 30
            return label

    return label

def combine_label(label):
    h, w = label.shape
    for j in range(w):
        for i in range(h):
            label[i][j] = convertDict[label[i][j]]
    return label


def changelabel(dirname):
    anno_dir = os.path.join(dirname, 'Parsing-final')
    anno_list = os.listdir(anno_dir)
    num_anno = len(anno_list)
    val_anno_list = random.sample(anno_list, num_anno // 10)
    trainList = open(os.path.join(dirname, 'trainList.txt'), 'w')
    valList = open(os.path.join(dirname, 'valList.txt'), 'w')

    if not os.path.exists(os.path.join(dirname, 'cam_a_seg')):
        os.makedirs(os.path.join(dirname, 'cam_a_seg'))
    if not os.path.exists(os.path.join(dirname, 'cam_b_seg')):
        os.makedirs(os.path.join(dirname, 'cam_b_seg'))

    for anno_name in tqdm(anno_list):
        anno = cv2.imread(os.path.join(anno_dir, anno_name), cv2.IMREAD_GRAYSCALE)
        anno = seperate_hat_bag(anno)
        anno = combine_label(anno)
        img_name = os.path.splitext(anno_name)[0][:-2] + '.bmp'
        cam_id = os.path.splitext(anno_name)[0].split('_')[-1]
        if cam_id == '1':
            cam = 'cam_a'
        elif cam_id == '2':
            cam = 'cam_b'
        else:
            raise ValueError
        name_img = os.path.join(cam, img_name)
        name_seg = os.path.join(cam + '_seg', img_name)
        path_seg = os.path.join(dirname, name_seg)
        cv2.imwrite(path_seg, anno)
        trainList.write(name_img + ' ' + name_seg + '\n')

        if anno_name in val_anno_list:
            valList.write(name_img + ' ' + name_seg + '\n')

    trainList.close()
    valList.close()


if __name__ == "__main__":
    dir_path = './data/viper'
    changelabel(dir_path)