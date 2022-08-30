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
names = os.listdir(img_root)

dest_root = args.dest_root + '/'
os.makedirs(args.dest_root, exist_ok=True)
os.makedirs(dest_root+'nc_instances', exist_ok=True)
count = 0

for name in names:
    image = np.array(Image.open(img_root+name)) #讀取細胞核與細胞質合起來的instances圖(顏色: 40, 20)
    index = name[:-4]
    mask_list = glob.glob(mask_root+index+"_*") #取得合成instances圖的各個細胞分割(102_1.bmp, 102_2.bmp...)
    
    for mask_name in mask_list: 
        mask = np.array(Image.open(mask_name))
        nc_image=cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=mask)
        name = mask_name.split('/')[-1]
        
        if np.all(nc_image == 0):
            print(mask_name)
            print('zero matrix')
            count = count + 1
            continue
        else:
            cv2.imwrite(dest_root + 'nc_instances/' + name, nc_image)
print(count)
    


