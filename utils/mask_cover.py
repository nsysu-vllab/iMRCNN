import argparse

args = argparse.ArgumentParser()
args.add_argument('--img_root',type=str,required=True,help="path to the folder where the images are saved")
args.add_argument('--mask_root',type=str,required=True,help="path to the folder where gt instances are saved")
args.add_argument('--dest_root',type=str,required=True,help="path to the folder where the COCO format json file and resized masks and images will be saved")

args = args.parse_args()

import glob
import os
import albumentations as albu
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import json
import argparse
import numpy as np

img_root = args.img_root + '/'
mask_root = args.mask_root + '/'
#找出 img_root 中所有圖片的號碼(ex: 102, 104...)
names = os.listdir(img_root)

dest_root = args.dest_root + '/'
os.makedirs(args.dest_root, exist_ok=True)
os.makedirs(dest_root+'overlapping', exist_ok=True)
green = np.array([100], dtype=np.uint8) 
yellow = np.array([150], dtype=np.uint8) 

for name in names:
    image = cv2.imread(img_root+name)#讀取細胞核與細胞質合起來的instances圖(顏色: 40, 20)
    mask = cv2.imread(mask_root+name)
    image[mask == 20] = green
    image[mask == 40] = yellow
    
    cv2.imwrite(dest_root+'overlapping/'+name, image)

#(2560, 1920)
#(1440, 1080)

#image = np.array(Image.open(img_root+name))
#image = np.array(Image.open('/home/frank/Desktop/instance segmentation/SegPC-2021-main/segment_result_ncCombine_val_4020/nc_combine/102.bmp'))
#mask = np.array(Image.open('/home/frank/Desktop/instance segmentation/SegPC-2021-main/segment_result/102_1.bmp'))

#image=cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=mask)

#cv2.imwrite('/home/frank/Desktop/instance segmentation/SegPC-2021-main/segment_result_ncCombine_val_4020/cell_image.bmp', image)

