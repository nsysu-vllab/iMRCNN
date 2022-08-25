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

#(2560, 1920)
#(1440, 1080)

inf = {
    "description": "SegPC ISBI Cell Segmentation Challenge",
    "url": "https://segpc-2021.grand-challenge.org/SegPC-2021/",
    "year": 2021,
    "date_created": "6th Feb,2021",
}

categories = [
    {"supercategory": "cell_st","id": 1,"name": "Nuclei"},
    {"supercategory": "cell_st","id": 2,"name": "Cytoplasm"},
]


img_root = args.img_root + '/'
mask_root = args.mask_root + '/'
names = os.listdir(img_root)


dest_root = args.dest_root + '/'
os.makedirs(args.dest_root, exist_ok=True)
#os.makedirs(dest_root+'x', exist_ok=True)
os.makedirs(dest_root+'instance_y', exist_ok=True)
os.makedirs(dest_root+'nc_combine', exist_ok=True)

images = []
annos = []
res_size=(1080,1440)

var= 1

for name in names:
    old_name = img_root + name
    name = name.split('.')[0]
    print(name)
    new_name = img_root + name + '000.bmp'
    os.rename(old_name, new_name)
    
    
    

