# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from skimage import measure
import numpy as np
import copy

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances, BitMasks, ROIMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.layers.roi_align import ROIAlign

__all__ = [
    "BaseMaskRCNNHead",
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "ROI_MASK_HEAD_REGISTRY",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@torch.jit.unused
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
    
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def mask_rcnn_loss(self, pred_mask_logits: torch.Tensor, instances: List[Instances], old_instances: List[Instances] = None, batched_inputs = None, images = None, vis_period: int = 0, o_pred_mask_logits = None, o_instances = None):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    """print("pred_mask_logits")
    print(pred_mask_logits.size())
    print("instances")
    print(instances)
    print("old_instances")
    print(old_instances)
    print("o_instances")
    print(o_instances)"""
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    old_total_num_masks = o_pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    #print(instances)
    gt_classes = []
    gt_masks = []
    old_gt_masks = []
    old_classes = []
    old_masks = []
    bit_masks_test = []
    #print("instances: ", instances.gt_masks)
    #print("old_instances: ", old_instances.pred_masks)
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)
        #print("instances_per_image: ",instances_per_image)
        #print("gt_masks: ", type(gt_masks_per_image))
    """
    for instances_per_image, old_ins_per_image in zip(instances, old_instances):
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        old_boxes = old_ins_per_image.proposal_boxes.tensor.detach()
        boxes = copy.deepcopy(old_boxes)
        
        old_gt_masks_per_image = instances_per_image.gt_masks.old_gt_crop_and_resize(
            boxes, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        old_gt_masks.append(old_gt_masks_per_image)
    print("gt_masks: ", gt_masks)
    print("old_gt_masks: ", old_gt_masks)
    """
    #KD loss change start
    #print('old_instances')
    #print(old_instances)
    if old_instances is not None:
        #conver pred_mask to polygons start
        for results, input_per_image, image_size in zip(
            old_instances, batched_inputs, images.image_sizes
        ):
            if len(results) == 0:
                continue
            #print('results_pred_masks')
            #print(results.pred_masks.size())
            boxes = results.proposal_boxes.tensor.detach()
            mask_size = mask_side_len
            #output_height = input_per_image.get("height", image_size[0])
            #output_width = input_per_image.get("width", image_size[1])
            #if results.has("pred_masks"):
            #    if isinstance(results.pred_masks, ROIMasks):
            #        roi_masks = results.pred_masks
            #    else:
                    # pred_masks is a tensor of shape (N, 1, M, M)
            #        roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
            #    results.pred_masks = roi_masks.to_bitmasks(
            #        results.pred_boxes, output_height, output_width, mask_threshold
            #    )  # TODO return ROIMasks/BitMask object in the future
            
            old_masks_per_image = results.pred_masks.crop_and_resize(
                boxes, mask_size
            ).to(device=pred_mask_logits.device)
            old_masks.append(old_masks_per_image)
        #conver pred_mask to polygons end
        
        for old_instances_per_image in old_instances:
            if len(old_instances_per_image) == 0:
                print('old_instaces_nothing======================================')
                continue
            if not cls_agnostic_mask:
                old_classes_per_image = old_instances_per_image.pred_classes.to(dtype=torch.int64)
                old_classes.append(old_classes_per_image)
            
            #old_masks_per_image = instances_per_image.gt_masks.crop_and_resize(  #會出錯可是因為這的的self是instances_per_image的數量
            #    old_instances_per_image.proposal_boxes.tensor.detach(), mask_side_len
            #).to(device=pred_mask_logits.device)
            device = old_instances_per_image.pred_masks.device
            boxes = old_instances_per_image.proposal_boxes.tensor.detach()
            mask_size = mask_side_len
            #binary_masks = old_instances_per_image.pred_masks
            #binary_masks = binary_masks.to('cpu').detach().numpy()
            
            
            
            #print(len(boxes))
            batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
            rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5
            #print("rois")
            #print(rois.size())

            bit_masks = old_instances_per_image.pred_masks.to(dtype=torch.float32)
            #bit_masks = bit_masks.squeeze(dim = 1)
            rois = rois.to(device=device)
            #output = (
            #    ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            #    .forward(bit_masks, rois)
            #    .squeeze(1)
            #)
            #old_masks_per_image = output >= 0.5
            # A tensor of shape (N, M, M), N=#old_instances in the image; M=mask_side_len
            bit_masks_test.append(old_masks_per_image)
        
    #KD loss change end

    if len(gt_masks) == 0 and len(old_masks) == 0:
        return pred_mask_logits.sum() * 0
    if len(gt_masks) == 0:
        print('no gt_masks but have old_masks')
        return (pred_mask_logits.sum() * 0)
        #return (pred_mask_logits.sum() * 0) + 0.1*torch.mean(torch.frobenius_norm(pred_mask_logits[gt_masks.size(0):] - o_features, dim=-1))# + F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")
    if len(old_masks) != 0:
        old_masks = cat(old_masks, dim=0)
        old_masks = old_masks.to(dtype=torch.float32)
    gt_masks = cat(gt_masks, dim=0)
    
    ##################
    #在下面的地方挑出正確的類別 再更後面的地方做sigmoid
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        o_pred_mask_logits = o_pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        o_indices = torch.arange(old_total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        #print('old_classes_none')
        #print(old_classes)
        #print(type(old_classes))
        if old_classes:
            old_classes = cat(old_classes, dim=0)
            #print('indices', indices)
            #print('old_classes',old_classes.size())
            #print('gt_classes', gt_classes.size())
            classes = torch.cat((gt_classes, old_classes), 0)
            pred_mask_logits = pred_mask_logits[indices, classes]
            #o_pred_mask_logits = np.squeeze(o_pred_mask_logits)
            o_pred_mask_logits = o_pred_mask_logits[o_indices, old_classes]
        else:
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)
    
    #print('old_masks',old_masks)
    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits[0:gt_masks.size(0)] > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)
    #print("pred_mask_logits", pred_mask_logits.size())
    #print("gt_masks", gt_masks.size(0))
    #print("old_logits", pred_mask_logits[gt_masks.size(0):].size())
    #print("gt_logits", pred_mask_logits[0:gt_masks.size(0)].size())
    #print("pred_mask_logits[0:gt_masks.size(0)]", pred_mask_logits[0:gt_masks.size(0)].size())
    #print("gt_masks", gt_masks.size())
    
    if len(old_masks) == 0:
        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean")
    else:
        #MiB loss start old_masks修改成o_features看看
        
        #pred_ml = pred_mask_logits[gt_masks.size(0):]
        #inputs = pred_ml.narrow(1, 0, old_masks.shape[1])
        #o_pred_ml = o_pred_mask_logits
        #o_inputs = o_pred_ml.narrow(1, 0, old_masks.shape[1])
        
        outputs = pred_mask_logits[gt_masks.size(0):].sigmoid()
        o_outputs = o_pred_mask_logits.sigmoid()
        
        #outputs = torch.log_softmax(inputs, dim=1)
        #o_outputs = torch.softmax(o_inputs, dim=1)
        #labels = torch.softmax(old_masks * 1, dim=1)
        
        KD_loss = (o_outputs * torch.log(outputs + 1e-8)) + ((1-o_outputs)*torch.log(1-outputs + 1e-8))
        #KD_loss = (o_outputs * torch.log(outputs)) + ((1-o_outputs)*torch.log(1-outputs))
        
        #test_out = pred_mask_logits[0:gt_masks.size(0)]
        #print("gt_masks: ", gt_masks)  =======================================================================================================================================
        #print("test_out: ", test_out)
        #mul_out = test_out * gt_masks
        #print("mul_out: ", mul_out)
        #KD_loss = (o_outputs * (torch.log(torch.abs(o_outputs-outputs)))) + ((1-o_outputs)*(torch.log(torch.abs((1-outputs)-(1-o_outputs)))))
        #print("first loss")
        #print(torch.log(torch.abs(o_outputs-outputs)))
        #print("second loss")
        #print(torch.log(torch.abs((1-outputs)-(1-o_outputs))))
        """
        # sigmoid
        prob = pred_mask_logits[gt_masks.size(0):].sigmoid()
        o_prob = o_pred_mask_logits.sigmoid()
        #print("prob")
        #print(prob)
        #print("o_prob")
        #print(o_prob)
        #KD_loss =  o_prob*(torch.log(o_prob)-torch.log(prob)) + (1-o_prob)*(torch.log(1-o_prob)-torch.log(1-prob))
        KD_loss =  o_prob*torch.log(prob) + (1-o_prob)*torch.log(1-prob)
        """
        
        print("mask_head323 outKD", self.outKD)
        
        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean")
        if self.outKD:
            mask_loss = mask_loss + 30*(-torch.mean(KD_loss))
            #mask_loss = mask_loss + 20*(-torch.mean(KD_loss))    #multi step KD的step3要降低權重
        if self.PseudoLabel:
            mask_loss = mask_loss + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")
        
        
        
        #### only use KD, test weight = 0.3; old weight  = 0.1
        #mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 1*torch.mean(torch.frobenius_norm(pred_mask_logits[gt_masks.size(0):] - o_features, dim=-1))
        #mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 1*torch.mean(torch.norm(pred_mask_logits[gt_masks.size(0):] - o_features, dim=-1, p=2))
        
        #### 原本的loss + KDmib, test weight = ; old weight  = 10
        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 30*(-torch.mean(KD_loss)) #大使用30
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 30*F.binary_cross_entropy(outputs, o_outputs, reduction="mean")""" #sigmoid
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 20*(-torch.mean(KD_loss))""" #multi step KD的step3要降低權重
        
        #### Pseudo label + KL divergence;  "+ torch.mean(F.kl_div(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction='batchmean'))"
        # " + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")"
        #print('kl loss change')
        #mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 30*(-torch.mean(KD_loss))
        
        #### 原本的loss + Pseudo label + KDmib
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 10*(-torch.mean(KD_loss)) + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")""" # original
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 10*(-torch.mean(KD_loss)) + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")"""
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 0.5*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], o_pred_mask_logits, reduction="mean") + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")"""
        
        #### 原本的loss + Pseudo label + KDplop
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 0.1*torch.mean(torch.frobenius_norm(pred_mask_logits[gt_masks.size(0):] - o_features, dim=-1))  + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")"""
        
        #### 原本的loss + Pseudo label
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean") + 1*F.binary_cross_entropy_with_logits(pred_mask_logits[gt_masks.size(0):], old_masks, reduction="mean")
        """
        #### 只有原本的loss
        """mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits[0:gt_masks.size(0)], gt_masks, reduction="mean")"""
    
    return mask_loss


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0, outKD, feaKD, PseudoLabel):
        """
        NOTE: this interface is experimental.

        Args:
            loss_weight (float): multiplier of the loss
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.outKD = outKD
        self.feaKD = feaKD
        self.PseudoLabel = PseudoLabel
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD,
                "outKD": cfg.outKD,
                "feaKD": cfg.feaKD,
                "PseudoLabel": cfg.PseudoLabel,
               }

    def forward(self, x, instances: List[Instances], old_instances: List[Instances] = None, batched_inputs = None, images = None, o_features = None, o_instances = None):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training and old_instances is not None:
            return {"loss_mask": mask_rcnn_loss(self, x, instances, old_instances, batched_inputs, images, self.vis_period, o_features, o_instances)* self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
