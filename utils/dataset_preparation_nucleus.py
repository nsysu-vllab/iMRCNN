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

    semantic_mask1 = np.zeros(res_size) #一個紀錄細胞質
    semantic_mask2 = np.zeros(res_size) #一個紀錄細胞核

    mask_list = glob.glob(mask_root+index+"_*")
    count = 0
    ins_img = glob.glob(dest_root+'instance_y/'+index+'_*')
    if (len(ins_img) > 0):
        print('ins_img num')
        print(len(ins_img))
        ins_count = len(ins_img)
    else:
        ins_count = 0 #紀錄有幾張instance的分割圖片
    for mask_name in mask_list: 
        count+=1
        
        total_mask = cv2.imread(mask_name, 0) #gray scale
        total_mask= cv2.resize(total_mask, res_size[::-1], interpolation=cv2.INTER_NEAREST)

        #semantic_mask = np.maximum(semantic_mask,total_mask)

        #將細胞質與細胞核分割開來
        print(np.unique(total_mask))
        color = np.unique(total_mask)
        for i in range(1):
            ann = {}
            ins_count+=1
            #mask_id = mask_name.split('/')[-1][:-6]
            #mask_id = mask_id+'_'+str(ins_count)
            
            if mask_name.split('/')[-1][-6] is '_' :
                mask_id = mask_name.split('/')[-1][:-6]
                mask_id = mask_id+'_'+str(ins_count)
            else:
                mask_id = mask_name.split('/')[-1][:-6]
                mask_id = mask_id+str(ins_count)

            mask = total_mask.copy()
            mask[mask == color[1]] = 0
            img.imsave(dest_root+'instance_y/'+mask_id+'.bmp', mask)
            
            semantic_mask1 = np.maximum(semantic_mask1, mask)

            bin_mask = np.zeros(mask.shape)
            bin_mask[mask>0] = 1

            res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   
            
            # ann = create_sub_mask_annotation(sub_mask = bin_mask, image_id = index, category_id=1, annotation_id = mask_id , is_crowd = 0)

            ann['id'] = mask_id
            print(mask_id)
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
            for r in res[0]:
                if r.shape[0]>mx:
                    mx = r.shape[0]
                    a = r
            ann['area'] =  cv2.contourArea(a)
            print(ann['area'])
            a = a.squeeze()
            print(a.shape)
            max_x, max_y = np.max(a, axis =0)
            min_x, min_y = np.min(a, axis =0)
            seg = a.ravel()
            seg = seg.astype('float64')
            ann['segmentation'].append(seg.tolist())

            ann["bbox"] =  [float(min_x-0.5), float(min_y)-0.5, float(max_x-min_x+1), float(max_y-min_y+1)]  #remove the -0.5 from the first and second item

            ann["iscrowd"]= 0
            ann["category_id"] = 1
            annos.append(ann)
    
        semantic_mask1 = (semantic_mask1>0)*255
        #bin_mask = (bin_mask>1)*255
        name_1 = name[:-4]+'_2'
        cv2.imwrite(dest_root+'semantic_y/'+name_1+'.bmp', semantic_mask1)
        
        #semantic_mask2 = (semantic_mask2>0)*255
        #bin_mask = (bin_mask>1)*255
        #name_2 = name[:-4]+'_2'
        #cv2.imwrite(dest_root+'semantic_y/'+name_2+'.bmp', semantic_mask2)

    print(count,"masks read")


dataset = {
    "info": inf,
    "licenses": [],
    "images": images,
    "annotations": annos,
    "categories": categories,
}

print("good_head")
with open(dest_root+'cCOCO.json', 'w') as fp:
    json.dump(dataset, fp)
    print("good")

print("number of images saved: ", os.listdir(dest_root+'x'))
print("number of instances saved: ", os.listdir(dest_root+'instance_y'))

