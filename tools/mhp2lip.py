import os
from collections import defaultdict
import cv2
import numpy as np


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
convertList = [0, 6, 6, 7, 1, 8, 8, 8, 8, 2,
               2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
               3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               9, 9, 9, 9, 2, 2, 2, 2, 2, 6,
               5, 0, 0, 2, 0, 5, 5, 7, 7, 8,
               2, 0, 0, 8, 8, 2, 0, 2, 3]


def name2dict(dirname):
    splits = [
        'train',
        'val'
    ]
    name_dict = defaultdict()
    for split in splits:
        split_dict = defaultdict(list)
        dir_path = os.path.join(dirname, split, "parsing_annos")
        name_list = os.listdir(dir_path)
        for name in name_list:
            img_idx = name.split('_')[0]
            split_dict[img_idx].append(os.path.join(dir_path, name))
        name_dict[split] = split_dict
    return name_dict


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


def img2box(dirname, name_dict):
    splits = [
        'train',
        'val'
    ]
    for split in splits:
        img_list = os.path.join(dirname, 'list', split+'.txt')
        splitList = open(os.path.join(dirname, split, split+'List.txt'), 'w')
        with open(img_list, 'r') as file:
            for img_idx in file:
                img_idx = img_idx.rstrip('\n')
                img = cv2.imread(os.path.join(dirname, split, 'images', img_idx+'.jpg'),
                                 cv2.IMREAD_COLOR)
                for path in name_dict[split][img_idx]:
                    name = path.split('/')[-1]
                    anno = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, -1]
                    bbox = find_box(anno, delta=4)
                    if (bbox[0] == bbox[2]) or (bbox[1] == bbox[3]):
                        continue
                    elif (bbox[2] - bbox[0]) * 1.5 > (bbox[3] - bbox[1]):
                        continue
                    name_seg = os.path.join('crop_seg', name)
                    name_img = os.path.join('crop_img', name).replace('png', 'jpg')
                    path_seg = path.replace('parsing_annos', 'crop_seg')
                    path_img = path.replace('parsing_annos', 'crop_img').replace('png', 'jpg')
                    crop_and_save(anno, bbox, path_seg)
                    crop_and_save(img, bbox, path_img)
                    splitList.write(name_img + ' ' + name_seg + '\n')
        splitList.close()


if __name__ == "__main__":
    dir_path = './data/LV-MHP-v2'
    name_dict = name2dict(dir_path)
    img2box(dir_path, name_dict)