# iMRCNN: Continual Instance Segmentation

## Data preparation

Download [SegPC](https://ieee-dataport.org/open-access/segpc-2021-segmentation-multiple-myeloma-plasma-cells-microscopic-images) Dataset
```
dataset_preparation_all.py: Generate a dataset on the separation of nucleus and cytoplasm.
dataset_preparation_cytoplasm.py: Generate a cytoplasm-only dataset.
dataset_preparation_nucleus.py: Generate a nucleus-only dataset.
```
Example commands of generating a nucleus-only dataset
```
python3 utils/dataset_preparation_nucleus.py 
--img_root {path of the image set used to generate the dataset}
--mask_root  {path of the ground truth set used to generate the dataset}
--dest_root {path of the dataset saved}

```
Example:
```
python3 utils/dataset_preparation_nucleus.py 
--img_root /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/train_step2 
--mask_root /home/frank/Desktop/instance\ segmentation/Pytorch-UNet-master/data/bsalf_mask 
--dest_root /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/bsalf_nuclei

```
## Installation

Detectron2 installation:

See [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

## Environmet

mrcnn (original detectron2):
```
# create conda environment
conda create -n mrcnn python=3.7
conda activate mrcnn
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cat Cascade_Mask_RCNN_X152/requirements.txt | xargs -n 1 pip3 install
```

imrcnn (modified detectron2):
```
# create conda environment
conda create -n imrcnn python=3.7
conda activate imrcnn
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cat Cascade_Mask_RCNN_X152/requirements.txt | xargs -n 1 pip3 install
```

## Training 

mrcnn (original detectron2)

step1:
```
python3 Cascade_Mask_RCNN_X152/CMRCNN_X152_train.py
--backbone Original 
--train_data_root  {path of training set}
--training_json_path  {path of COCO.json of training set}
--val_data_root  {path of validation set}
--validation_json_path {path of COCO.json of validation set}
--work_dir {path of model saved}

```
Example:
```
python3 Cascade_Mask_RCNN_X152/CMRCNN_X152_train.py 
--backbone Original 
--train_data_root /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/fsalf_nuclei/x
--training_json_path /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/fsalf_nuclei/nCOCO.json
--val_data_root /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/bsalf_nuclei/x
--validation_json_path /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/bsalf_nuclei/nCOCO.json
--work_dir /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/model_and_log/model_and_log_step1
```

imrcnn (modified detectron2):

step2:
```
python3 Cascade_Mask_RCNN_X152/imrcnn_cascade.py
--backbone Original 
--train_data_root {path of training set}
--training_json_path {path of COCO.json of training set}
--val_data_root {path of validation set}
--validation_json_path {path of COCO.json of validation set}
--work_dir {path of model saved}
--weight_folder {path of pretrained model}
--weight_file  {file of pretrained model}
--num_class {number of classes}
--outKD --feaKD --PseudoLabel  {components that you can add in}
```
Example:
```
python3 Cascade_Mask_RCNN_X152/imrcnn_cascade.py 
--backbone Original 
--train_data_root /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/bsalf_cytoplasm/x
--training_json_path /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/bsalf_cytoplasm/cCOCO.json
--val_data_root /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/fsalf_cytoplasm/x
--validation_json_path /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/data/fsalf_cytoplasm/cCOCO.json
--work_dir /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/model_and_log/ model_and_log_step2
--weight_folder /home/frank/Desktop/instance\ segmentation/SegPC-2021-main/model_and_log/model_and_log_nuclei_CASCADE/
--weight_file model_final.pth
--num_class 2
--outKD --feaKD --PseudoLabel

```

## Inference
Generate the segment prediction images of the model with CMRCNN_X152_inference.py
```
python3 Cascade_Mask_RCNN_X152/CMRCNN_X152_inference.py
--backbone Original 
--saved_model_path {path of the model which you mant to use}
--input_images_folder {folder of the image set which you use as inputs and predictions}
--save_path {path of prediction results}
```
Combine the nucleus and cytoplasm which was separated from the same image with utils/nc_combine.py
```
python3 utils/nc_combine.py
--img_root {path of the testing set}
--mask_root {path of the prediction result of nucleus and cytoplasm}
--dest_root {path of result saved}
```
Segment each cell with the mask which is the segmentation of whole cell
```
python3 utils/cell_mask.py
--img_root {path of the combination of nucleus and cytoplasm}
--mask_root {path of the segmentation of whole cell}
--dest_root {path of the final result}

```

## Evaluate
Use a .txt file which was generated by submission.py as inputs of evaluate.py
```
python3 SegPC_mIoU_evaluator/sub_and_eval_original/submission.py
-s {the prediction result of model}
-d {path of submission.txt saved}
```
Show mIou with evaluate.py
```
python3 SegPC_mIoU_evaluator/sub_and_eval_original/evaluate.py
```

## Acknowledgement

Our implementation is based on these repositories: [SegPC-2021](https://github.com/dsciitism/SegPC-2021)





