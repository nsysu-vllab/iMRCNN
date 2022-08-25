import numpy as np
import os
import glob
import cv2
import numba
import tqdm
from numba import jit
import platform

delim = '\\' if platform.system() == 'Windows' else '/'

#gt_path = './gt_rez_ins'
#gt_path = './data/nc_seperate_val_blackwhite/instance_y'
gt_path = '/home/frank/Desktop/instance segmentation/SegPC-2021-main/data/val_gt_cy/instance_y'    #### 要單獨測試cy時使用 val_gt_cy;  要單獨測試nu時使用 val_gt_nu
#gt_path = '/home/frank/Desktop/instance segmentation/SegPC-2021-main/data/new_validation/BW_instance'



global res
res = (1080,1440)
imgs_gt = list(set([i.split(delim)[-1].split('_')[0] for i in glob.glob(gt_path+'/*')]))
imgs_gt = sorted(imgs_gt)
ins_gt = {}

for img in imgs_gt:
    ins_gt[img] = np.float32([cv2.resize(cv2.imread(ins,0).astype(np.float32),res[::-1],interpolation=cv2.INTER_NEAREST) for ins in glob.glob(gt_path+'/'+img+'_*')])

#@jit(nopython=True)
def iou(x,y):
    insec = np.logical_and(x,y)
    uni = np.logical_or(x,y)
    return np.sum(insec)/(np.sum(uni))


def getScore(gt_,ins_pred):
    pred = np.zeros(res,dtype=np.bool_)
    result = 0.0
    for i in range(gt_.shape[0]):
        mask_gt = gt_[i] > 0
        iou_ = 0.0
        for ins_p in ins_pred:
            #print(ins_p[0] == [['']])
            #if ins_p[0] != [['']]:
            #    for m,n in ins_p[0]:
            #        try:
            #            pred[int(m),int(n)] = True
            #        except:
            #            pass
            if ins_p[1] != [['']]:
                for m,n in ins_p[1]:
                    try:
                        pred[int(m),int(n)] = True
                    except:
                        pass

            tm_iou = iou(mask_gt,pred)
            pred[:,:] = False
            if tm_iou > iou_:
                iou_ = tm_iou
        result += iou_
    return result

def calcScore(submission):
    score = 0.0
    cnt = 0
    print(ins_gt.keys())
    print('submission')
    print(submission.keys())
    #print('img')
    for img in tqdm.tqdm(imgs_gt):
        img = str(img)
        #print(img, end = ',')
        try:
            #print('1111')
            score += getScore(ins_gt[img],submission[img])
            #print(score, end=',')
        except:
            #print('2222')
            pass # img absent in prediction
        cnt += ins_gt[img].shape[0]
    #print('ins_gt')
    #print(ins_gt['102'])
    #score += getScore(ins_gt['102'], submission['102'])
    #print(submission['102'][0])
    #print(submission['102'][1])
    return score/cnt


    


def parse(submission):
    with open(submission,'r') as fp:
        data = fp.read().split('\n')
    print(len(data))
    for d in tqdm.tqdm(data):
        ins = d.split('\t')
        #print('ins')
        #print(ins[1])
        #print('ins2')
        #print(ins[2])
        #print('parse')
        #print(ins)
        yield ins[0],[[[[l for l in k.split(',')] for k in j.split(';')] for j in i.split(' ')] for i in ins[1:]]
        #yield ins[0],[[[[l for l in k.split(',')] for k in i.split(';')]] for i in ins[1:]]
    


def evaluate(submission):
    sub = {}
    for im,ins in parse(submission):
        if im in imgs_gt:
            sub[im] = ins
            #print('ins')
            #print(ins)
    return calcScore(sub)

if __name__ == '__main__':
    sub = './submission.txt'
    #sub = './model_and_log/model_and_log_step2_KDmib/segment_result_ncCombine_val_4020/submission_nu.txt'
    score = evaluate(sub)
    print(score)




