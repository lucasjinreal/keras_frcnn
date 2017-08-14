"""
this file will load all kitti original label file into
a single file from which we can index every image and its
bounding boxes

Usage:

python2 generate_simple_kitti_anno_file.py \
/media/jintian/Netac/Datasets/Kitti/object/training/image_2 \
/media/jintian/Netac/Datasets/Kitti/object/training/label_2
"""
from __future__ import print_function, division
import numpy as np
import os
import sys


def generate(img_dir_, label_dir_):
    """
    convert kitti data into a single txt file, with this format:
    Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01

    type, truncated, occluded, alpha,
    :param img_dir_:
    :param label_dir_:
    :return:
    """

    if not os.path.exists(label_dir_):
        print('label dir: {} doest not exist'.format(label_dir_))
        exit(0)
    all_label_files = [i for i in os.listdir(label_dir_) if i.endswith('.txt')]
    print('got {} label files.'.format(len(all_label_files)))

    all_img_lables = []

    target_file = open('kitti_simple_label.txt', 'w')
    for label_file_name in all_label_files:
        label_file = os.path.join(label_dir_, label_file_name)

        with open(label_file, 'r') as f:
            for l in f.readlines():
                class_name, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = l.strip().split(' ')
                target_file.write('{},{},{},{},{},{}\n'.format(
                    os.path.join(img_dir_, label_file_name.replace('txt', 'png')),
                                                               x1, y1, x2, y2, class_name))

    target_file.close()
    print('convert finished.')


if __name__ == '__main__':
    img_dir = sys.argv[1]
    label_dir = sys.argv[2]
    generate(img_dir, label_dir)
