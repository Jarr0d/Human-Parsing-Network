import os
from collections import defaultdict
import cv2
import numpy as np
import random
from tqdm import tqdm


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
convertList = [0, 6, 1, 7, 2, 3, 3, 2, 3, 4,
               4, 7, 9, 9, 8, 8, 5, 2]


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


def crop_and_save(img, bbox, savedir):
    x1, y1, x2, y2 = bbox
    dim = img.ndim
    if dim == 2:
        cropped = img[y1:y2 + 1, x1:x2 + 1]
        cropped = combine_label(cropped)
    else:
        cropped = img[y1:y2 + 1, x1:x2 + 1, :]
    cv2.imwrite(savedir, cropped)


def combine_label(label):
    h, w = label.shape
    for j in range(w):
        for i in range(h):
            label[i][j] = convertList[label[i][j]]
    return label


def img2box(dirname):
    img_dir = os.path.join(dirname, 'JPEGImages')
    anno_dir = os.path.join(dirname, 'SegmentationClassAug')
    img_list = os.listdir(img_dir)
    num_img = len(img_list)
    val_img_list = random.sample(img_list, num_img // 10)
    trainList = open(os.path.join(dirname, 'trainList.txt'), 'w')
    valList = open(os.path.join(dirname, 'valList.txt'), 'w')

    if not os.path.exists(os.path.join(dirname, 'crop_img')):
        os.makedirs(os.path.join(dirname, 'crop_img'))
    if not os.path.exists(os.path.join(dirname, 'crop_seg')):
        os.makedirs(os.path.join(dirname, 'crop_seg'))

    for img_name in tqdm(img_list):
        img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_COLOR)
        anno_name = img_name.replace('jpg', 'png')
        anno = cv2.imread(os.path.join(anno_dir, anno_name), cv2.IMREAD_GRAYSCALE)
        bbox = find_box(anno, delta=4)
        if (bbox[0] == bbox[2]) or (bbox[1] == bbox[3]):
            continue
        name_img = os.path.join('crop_img', img_name)
        name_seg = os.path.join('crop_seg', anno_name)
        path_img = os.path.join(dirname, name_img)
        path_seg = os.path.join(dirname, name_seg)
        crop_and_save(img, bbox, path_img)
        crop_and_save(anno, bbox, path_seg)
        trainList.write(name_img + ' ' + name_seg + '\n')

        if img_name in val_img_list:
            valList.write(name_img + ' ' + name_seg + '\n')

    trainList.close()
    valList.close()


if __name__ == "__main__":
    dir_path = './data/atr'
    img2box(dir_path)