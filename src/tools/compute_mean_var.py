#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :compute_mean_var.py
@brief       :计算所有图片的均值和方差
@time        :2020/04/12 10:51:53
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :python3 compute_mean_var.py -i image_dir
'''

import os
from imutils.paths import list_images
import numpy as np
import argparse
from numba import jit
import cv2

def compute(image_dir):
    img_lists = list(list_images(image_dir))

    R_channel = 0
    G_channel = 0
    B_channel = 0
    total_pixel = 0
    for i in img_lists:
        img = cv2.imread(i)
        height , width = img.shape[:2]

        total_pixel += (height * width)

        R_channel = R_channel + np.sum(img[: , : , 2])
        G_channel = G_channel + np.sum(img[: , : , 1])
        B_channel = B_channel + np.sum(img[: , : , 0])

    R_mean = R_channel/total_pixel
    G_mean = G_channel/total_pixel
    B_mean = B_channel/total_pixel

    R_channel = 0
    G_channel = 0
    B_channel = 0

    for i in img_lists:
        img = cv2.imread(i)

        R_channel = R_channel + np.sum((img[: , : , 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[: , : , 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[: , : , 2] - B_mean) ** 2)
    

    R_var = np.sqrt(R_channel / (total_pixel * 255 * 255))
    G_var = np.sqrt(G_channel / (total_pixel * 255 * 255))
    B_var = np.sqrt(B_channel / (total_pixel * 255 * 255))

    R_mean = R_mean/255
    G_mean = G_mean/255
    B_mean = B_mean/255

    print(f"mean: [{R_mean} , {G_mean} , {B_mean}] , var: [{R_var} , {G_var} , {B_var}]")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' , '--images' , help='input image dir')

    args = vars(parser.parse_args())
    input_image_dir = args['images']

    compute(input_image_dir)




