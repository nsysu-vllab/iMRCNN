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
    {"supercategory": "cell_st","id": 1,"name": "cell"},
]


img_root = args.img_root + '/old_instance_y/'
mask_root = args.mask_root + '/'
names = os.listdir(img_root)


dest_root = args.dest_root + '/'
os.makedirs(args.dest_root, exist_ok=True)
os.makedirs(dest_root+'x', exist_ok=True)
os.makedirs(dest_root+'instance_y', exist_ok=True)
#os.makedirs(dest_root+'semantic_y', exist_ok=True)

images = []
annos = []
res_size=(1080,1440)

var= 1

for name in names:
    print(var)
    var+=1
    print(name)
    image = np.array(Image.open(img_root+name))

    image= cv2.resize(image, res_size[::-1],interpolation=cv2.INTER_NEAREST)
    new_im = Image.fromarray(image)
    #new_im.save(dest_root+'x/'+name)
    # print(image.shape)

    h,w,_ = image.shape
    index = name[:-4]
    
    for i in range(h):
        for j in range(w):
            #print(image[i, j])
            if (image[i, j] == [253, 231, 36]).all():
                image[i, j] = [0,143,139]

    img.imsave(dest_root+'instance_y/'+name, image)
    
    #===============================

