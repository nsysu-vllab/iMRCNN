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

    img_info = {}
    img_info['file_name'] = name
    img_info['height'] = h
    img_info['width'] = w
    img_info['id'] = int(index)
    images.append(img_info)

    semantic_mask1 = np.zeros(res_size) #一個紀錄細胞質
    semantic_mask2 = np.zeros(res_size) #一個紀錄細胞核

    mask_list = glob.glob(mask_root+index+"_*")
    count = 0
    ins_count = 0 #紀錄有幾張instance的分割圖片
    for mask_name in mask_list: 
        count+=1
        
        total_mask = cv2.imread(mask_name, 0) #gray scale
        total_mask= cv2.resize(total_mask, res_size[::-1], interpolation=cv2.INTER_NEAREST)

        #semantic_mask = np.maximum(semantic_mask,total_mask)

        #將細胞質與細胞核分割開來
        print(np.unique(total_mask))
        color = np.unique(total_mask)

        ann = {}
        ins_count+=1
        if mask_name.split('/')[-1][-6] is '_' :
            mask_id = mask_name.split('/')[-1][:-6]
            mask_id = mask_id+'_'+str(ins_count)
        else:
            mask_id = mask_name.split('/')[-1][:-6]
            mask_id = mask_id+str(ins_count)

        mask = total_mask.copy()
        #mask[mask == color[i]] = 0
        #ins_mask = np.zeros(res_size)
        #ins_mask = np.maximum(ins_mask, mask)
        #ins_mask = (ins_mask>0)*255
        #cv2.imwrite(dest_root+'instance_y/'+mask_id+'.bmp', ins_mask)
        #img.imsave(dest_root+'instance_y/'+mask_id+'.bmp', mask)
    
        #if i==1:
        semantic_mask1 = np.maximum(semantic_mask1, mask)
        #if i==2:
        #    semantic_mask2 = np.maximum(semantic_mask2, mask)

        bin_mask = np.zeros(mask.shape)
        bin_mask[mask>0] = 1
    
        #semantic_mask1 = (semantic_mask1>0)*255
        #bin_mask = (bin_mask>1)*255
        name_1 = name[:-4]
        cv2.imwrite(dest_root+'nc_combine/'+name_1+'.bmp', semantic_mask1)
        
        #semantic_mask2 = (semantic_mask2>0)*255
        #bin_mask = (bin_mask>1)*255
        #name_2 = name[:-4]+'_2'
        #cv2.imwrite(dest_root+'semantic_y/'+name_2+'.bmp', semantic_mask2)

    print(count,"masks read")


#dataset = {
#    "info": inf,
#    "licenses": [],
#    "images": images,
#    "annotations": annos,
#    "categories": categories,
#}

print("good_head")
#with open(dest_root+'COCO.json', 'w') as fp:
#    json.dump(dataset, fp)
#    print("good")

print("number of images saved: ", os.listdir(dest_root+'x'))
print("number of instances saved: ", os.listdir(dest_root+'instance_y'))

