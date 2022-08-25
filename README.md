# iMRCNN: Continual Instance Segmentation

## Data preparation

Download [SegPC](https://ieee-dataport.org/open-access/segpc-2021-segmentation-multiple-myeloma-plasma-cells-microscopic-images) Dataset

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





