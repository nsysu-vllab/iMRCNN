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


img_root = args.img_root + '/'
mask_root = args.mask_root + '/'
names = os.listdir(img_root)


dest_root = args.dest_root + '/'
os.makedirs(args.dest_root, exist_ok=True)
os.makedirs(dest_root+'x', exist_ok=True)
os.makedirs(dest_root+'instance_y', exist_ok=True)
os.makedirs(dest_root+'semantic_y', exist_ok=True)

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
    new_im.save(dest_root+'x/'+name)
    # print(image.shape)

    h,w,_ = image.shape
    index = name[:-4]

    img_info = {}
    img_info['file_name'] = name
    img_info['height'] = h
    img_info['width'] = w
    img_info['id'] = int(index)
    images.append(img_info)

    semantic_mask = np.zeros(res_size)

    mask_list = glob.glob(mask_root+index+"_*")
    count = 0
    ins_count = 0 #紀錄有幾張instance的分割圖片
    for mask_name in mask_list: 
        count+=1
        ins_count+=1
        ann = {}
        mask = cv2.imread(mask_name, 0) #gray scale
        mask= cv2.resize(mask, res_size[::-1], interpolation=cv2.INTER_NEAREST)

        semantic_mask = np.maximum(semantic_mask,mask)
        
        ins_mask = np.zeros(res_size)
        ins_mask = np.maximum(ins_mask,mask)
        ins_mask = (ins_mask>0)*255
        #將細胞質與細胞核分割開來
        #print(np.unique(mask))
        #color = np.unique(mask)
        #mask1 = 
        
        mask_id = mask_name.split('/')[-1][:-4]
        cv2.imwrite(dest_root+'instance_y/'+mask_id+'.bmp', ins_mask)

        bin_mask = np.zeros(mask.shape)
        bin_mask[mask>0] = 1

        res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # ann = create_sub_mask_annotation(sub_mask = bin_mask, image_id = index, category_id=1, annotation_id = mask_id , is_crowd = 0)

        ann['id'] = mask_id
        ann['image_id'] = int(index)
        ann['segmentation'] = []


        # yy, xx = np.where(bin_mask)


        # seg = np.c_[xx,yy].ravel()
        # seg = seg.astype('float64')
        # ann['segmentation'].append(seg.tolist())


        # min_x, max_x, min_y, max_y = min(xx), max(xx), min(yy), max(yy)

        print(mask_name, res[0][0].shape, len(res[0]))
        a = res[0][0]
        mx = 0
        for i in res[0]:
            if i.shape[0]>mx:
                mx = i.shape[0]
                a = i
        ann['area'] =  cv2.contourArea(a)
        #ann['area'] =  cv2.contourArea(mask)
        print(ann['area'])
        a = a.squeeze()
        print(a.shape)
        max_x, max_y = np.max(a, axis =0)
        min_x, min_y = np.min(a, axis =0)
        seg = a.ravel()
        seg = seg.astype('float64')
        print(seg.shape)
        #chand for testing
        #seg = mask.ravel()
        #print(mask.shape)
        #seg = seg.astype('float64')
        ann['segmentation'].append(seg.tolist())

        ann["bbox"] =  [float(min_x-0.5), float(min_y)-0.5, float(max_x-min_x+1), float(max_y-min_y+1)]  #remove the -0.5 from the first and second item

        ann["iscrowd"]= 0
        ann["category_id"] = 1

        annos.append(ann)
    
    semantic_mask = (semantic_mask>0)*255
    #bin_mask = (bin_mask>1)*255
    cv2.imwrite(dest_root+'semantic_y/'+name,semantic_mask)

    print(count,"masks read")


dataset = {
    "info": inf,
    "licenses": [],
    "images": images,
    "annotations": annos,
    "categories": categories,
}

print("good_head")
with open(dest_root+'COCO.json', 'w') as fp:
    json.dump(dataset, fp)
    print("good")

print("number of images saved: ", os.listdir(dest_root+'x'))
print("number of instances saved: ", os.listdir(dest_root+'instance_y'))

