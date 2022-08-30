import argparse

args = argparse.ArgumentParser()
args.add_argument('--backbone',type=str,required=True,choices=["Original","Effb5","Transformer_Effb5"],help="The backbone to be used from the given choices")
args.add_argument('--train_data_root',type=str,required=True,help="path to training data root folder")
args.add_argument('--training_json_path',type=str,required=True,help="path to the training json file in COCO format")

args.add_argument('--val_data_root',type=str,required=True,help="path to validation data root folder")
args.add_argument('--validation_json_path',type=str,required=True,help="path to validation json file in COCO format")

args.add_argument('--work_dir',type=str,required=True,help="path to the folder where models and logs will be saved")
args.add_argument('--iterations',type=int,default=6000) #20000 設5000有到.4093
args.add_argument('--batch_size',type=int,default=1)

# Incrementl learning components
args.add_argument('--outKD', help="using output-level KD loss", action="store_true")
args.add_argument('--feaKD', help="using feature-level KD loss", action="store_true")
args.add_argument('--PseudoLabel', help="using Pseudo labeling", action="store_true")

args.add_argument('--weight_folder', type=str,required=True, help='path to the folder where the pretrained model parameter file is (Use folder path, not file path!!!)')
args.add_argument('--weight_file', type=str,required=True, help='path to the pretrained model parameter file (Use file path!!!)')
args.add_argument('--num_class', type=int,required=True, help='number of class')

args = args.parse_args()

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import math
import pycocotools.mask as mask_util
from torch.nn import functional as F
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.config import configurable
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model,build_resnet_backbone,build_backbone
from detectron2.structures import ImageList
from detectron2.structures import Instances, BitMasks, ROIMasks
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from typing import List, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from torch import nn
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference



from detectron2.data.datasets import register_coco_instances
register_coco_instances("SegPC_train", {}, args.training_json_path, args.train_data_root)
register_coco_instances("SegPC_val", {}, args.validation_json_path, args.val_data_root)


train_meta = MetadataCatalog.get('SegPC_train')
val_meta = MetadataCatalog.get('SegPC_val')

train_dicts = DatasetCatalog.get("SegPC_train")
val_dicts = DatasetCatalog.get("SegPC_val")


import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import *
import torch
import torch.nn as nn



class Transformer_Encoder(VisionTransformer):
    def __init__(self, pretrained = False, pretrained_model = None, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12,
                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):

        super(Transformer_Encoder, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=1000, embed_dim=embed_dim, depth=depth,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                  drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)
        
        self.num_classes = 1
        self.dispatcher = {
            'vit_small_patch16_224': vit_small_patch16_224,
            'vit_base_patch16_224': vit_base_patch16_224,
            'vit_large_patch16_224': vit_large_patch16_224,
            'vit_base_patch16_384': vit_base_patch16_384,
            'vit_base_patch32_384': vit_base_patch32_384,
            'vit_large_patch16_384': vit_large_patch16_384,
            'vit_large_patch32_384': vit_large_patch32_384,
            'vit_large_patch16_224' : vit_large_patch16_224,
            'vit_large_patch32_384': vit_large_patch32_384,
            'vit_small_resnet26d_224': vit_small_resnet26d_224,
            'vit_small_resnet50d_s3_224': vit_small_resnet50d_s3_224,
            'vit_base_resnet26d_224' : vit_base_resnet26d_224,
            'vit_base_resnet50d_224' : vit_base_resnet50d_224,
        }
        self.pretrained_model = pretrained_model
        self.pretrained = pretrained
        if pretrained:
            self.load_weights()
        self.head = nn.Identity()
        self.encoder_out = [1,2,3,4,5]

        

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        features = []

        for i,blk in enumerate(self.blocks,1):
            x = blk(x)
            if i in self.encoder_out:
                features.append(x)

        for i in range(len(features)):
            features[i] = self.norm(features[i])

        return features

    def forward(self, x):

        features = self.forward_features(x)
        return features
    
    def load_weights(self):
        model = None
        try:
            model = self.dispatcher[self.pretrained_model](pretrained=True)
        except:
            print('could not not load model')
        if model == None:
            return
        # try:
        self.load_state_dict(model.state_dict())
        print("successfully loaded weights!!!")
        
        # except:
        #     print("Could not load weights. Parameters should match!!!")

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, META_ARCH_REGISTRY, ROI_HEADS_REGISTRY
from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn

@BACKBONE_REGISTRY.register()
class Effb5(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()

    encoder_name = 'timm-efficientnet-b5'
    in_channels = 3
    encoder_depth = 5
    encoder_weights = 'noisy-student'
    self.encoder = get_encoder(encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights)
    self.channels = self.encoder.out_channels
    self.conv = nn.ModuleList(
        [nn.Conv2d(self.channels[i],256,3,stride = 2, padding = 1) for i in range(len(self.channels))]
    )

    self.names = ["p"+str(i+1) for i in range(6)]
  def forward(self, image):
    
    features = self.encoder(image)
    out = {self.names[i]: self.conv[i](features[i]) for i in range(1, len(features))}

    return out

  def output_shape(self):
    out_shape = {self.names[i]: ShapeSpec(channels =256, stride = 2**(i+1)) for i in range(1, len(self.names))}

    return out_shape
    
    
from torchvision.transforms import Resize
from einops import rearrange

@BACKBONE_REGISTRY.register()
class Transformer_Effb5(Backbone):
    def __init__(self, cfg, input_shape):

        super().__init__()
        self.emb_dim = 768
        self.pretrained = True
        self.pretrained_trans_model = 'vit_base_patch16_384'
        self.patch_size = 16
        
        self.transformer = Transformer_Encoder(pretrained = True, img_size = 384, pretrained_model = self.pretrained_trans_model, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias = True)
        self.encoder_name = 'timm-efficientnet-b5'
        self.in_channels = 3
        self.encoder_depth = 5
        self.encoder_weights = 'noisy-student'
        
        self.conv_encoder = get_encoder(self.encoder_name,
                in_channels=self.in_channels,
                depth=self.encoder_depth,
                weights=self.encoder_weights)
        
        self.conv_channels = self.conv_encoder.out_channels
        self.conv_final = nn.ModuleList(
            [nn.Conv2d(self.conv_channels[i],self.emb_dim,3,stride = 2, padding = 1) for i in range(1,len(self.conv_channels))]
        )
        self.names = ["p"+str(i+2) for i in range(5)]
        self.resize =  Resize((384,384))
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
        self.Wk = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
        
        


    def forward(self, image):
    
      conv_features = self.conv_encoder(image)
      # print("initial shape of conv features:")
      # print([i.shape for i in conv_features])
      conv_features = conv_features[1:]
      for i in range(len(self.conv_final)):
          conv_features[i]  = self.conv_final[i](conv_features[i])


      # print("final shape of conv features:")
      exp_shape = [i.shape for i in conv_features]
      # print(exp_shape)
      
      

      transformer_features = self.transformer(self.resize(image))
      # print("shape of transformer features:")
      # print([i.shape for i in transformer_features])
      
      # _ , l, e = transformer_features[0].shape
      # _ , e, h, w = conv_features[0].shape

      features = self.project(conv_features, transformer_features)
      features = self.emb2img(features, exp_shape)

      # print("shape of final features:")
      # print([i.shape for i in features])
      out = {self.names[i]: features[i] for i in range(len(features))}

      return out
    
    def project(self, conv_features, transformer_features):

        features = []

        for i in range(len(conv_features)):

            t = transformer_features[i]
            x = rearrange(conv_features[i], 'b c h w -> b (h w) c') 
            xwq = self.Wq(x)
            twk = self.Wk(t)
            twk_T = rearrange(twk, 'b l c -> b c l')
            A = torch.einsum('bij,bjk->bik', xwq, twk_T).softmax(dim = -1)
            x += torch.einsum('bij,bjk->bik', A, t)
            features.append(x)

        return features
    
    def emb2img(self, features, exp_shape):

        for i, x in enumerate(features):
            B, P, E = x.shape             #(batch_size, latent_dim, emb_dim)
            x = x.transpose(1,2).reshape(B, E, exp_shape[i][2], exp_shape[i][3])
            features[i] = x

        return features
    
    def output_shape(self):
      out_shape = {self.names[i]: ShapeSpec(channels =768, stride = 2**(i+2)) for i in range(len(self.names))}

      return out_shape


from typing import Dict, List, Optional, Tuple

@META_ARCH_REGISTRY.register()
class KD_CRCNN(GeneralizedRCNN):
        
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        ocfg = get_cfg()
        ocfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
        ocfg.MODEL.WEIGHTS = os.path.join(args.weight_folder, args.weight_file)  # path to the model weight just trained
        ocfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        ocfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        #新增
        ocfg.outKD = False
        ocfg.feaKD = False
        ocfg.PseudoLabel = False
        
        old_model = DefaultPredictor(ocfg)
        

        images = self.preprocess_image(batched_inputs)
        old_features = old_model.model.backbone(images.tensor)
        om_proposals,_ = old_model.model.proposal_generator(images, old_features, None) 
        test_proposals = om_proposals
        old_instances = old_model.model.roi_heads._forward_box(old_features, om_proposals)
        old_instances = old_model.model.roi_heads.forward_with_given_boxes(old_features, old_instances)
        
        #INFERENCE
        for results, input_per_image, image_size in zip(
            old_instances, batched_inputs, images.image_sizes
        ):
            output_height = input_per_image.get("height", image_size[0])
            output_width = input_per_image.get("width", image_size[1])
            if results.has("pred_masks"):
                if isinstance(results.pred_masks, ROIMasks):
                    roi_masks = results.pred_masks
                else:
                    # pred_masks is a tensor of shape (N, 1, M, M)
                    roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
                results.pred_masks = roi_masks.to_bitmasks(
                    results.pred_boxes, output_height, output_width
                )  # TODO return ROIMasks/BitMask object in the future
        
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        new_oldMask = None
        for old_in in old_instances:
            for oldMask in old_in.pred_masks:
                h = oldMask.size(0)
                w = oldMask.size(1)
        
                for gt_in in gt_instances:
                    for masks in gt_in.gt_masks:
                        gtMask = polygons_to_bitmask(masks, h, w)
                        gtMask = torch.from_numpy(gtMask)
                        gtMask = gtMask.to(dtype=torch.float32)
                        # 0->1; 1->0;
                        gtMask = torch.where(gtMask > 0.5, 0, 1).to(self.device)
                        
                        old_Mask = oldMask*gtMask
                old_Mask = old_Mask.unsqueeze(0)
                
                if new_oldMask is None and old_Mask.any() is False:
                    new_oldMask = oldMask
                elif new_oldMask is None:
                    new_oldMask = old_Mask
                elif old_Mask.any() is False:
                    new_oldMask = tarch.cat((new_oldMask, oldMask), 0)
                else:
                    new_oldMask = torch.cat((new_oldMask, old_Mask),0) 
            if new_oldMask is None:
                break
            else:
                old_in.pred_masks = BitMasks(new_oldMask)
        
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            old_proposals, old_proposal_losses = self.proposal_generator(images, features, old_instances) 
            
            for k,v in old_proposal_losses.items():
                if k in proposal_losses.keys():
                    proposal_losses[k] += v
                else:
                    proposal_losses[k] = v
            
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
            
        _, detector_losses = self.roi_heads(images, features, proposals, old_features, old_proposals, om_proposals, old_model, batched_inputs, gt_instances, old_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
        
    
    def _local_pod(self, x, spp_scales=[1, 2], normalize=False, normalize_per_scale=False):
        b = x.shape[0]
        w = x.shape[-1]
        emb = []

        for scale_index, scale in enumerate(spp_scales):
            k = w // scale

            nb_regions = scale**2

            for i in range(scale):
                for j in range(scale):
                    tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                    horizontal_pool = tensor.mean(dim=3).view(b, -1)
                    vertical_pool = tensor.mean(dim=2).view(b, -1)

                    if normalize_per_scale is True:
                        horizontal_pool = horizontal_pool / nb_regions
                        vertical_pool = vertical_pool / nb_regions
                    elif normalize_per_scale == "spm":
                        if scale_index == 0:
                            factor = 2 ** (len(spp_scales) - 1)
                        else:
                            factor = 2 ** (len(spp_scales) - scale_index)
                        horizontal_pool = horizontal_pool / factor
                        vertical_pool = vertical_pool / factor

                    if normalize:
                        horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                        vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                    emb.append(horizontal_pool)
                    emb.append(vertical_pool)

        return torch.cat(emb, dim=1)

@ROI_HEADS_REGISTRY.register()
class KD_CROIHeads(CascadeROIHeads):
    def forward( self, images: ImageList, features: Dict[str, torch.Tensor], origin_proposals: List[Instances], old_features: Dict[str, torch.Tensor] = None, old_proposals: List[Instances] = None, om_proposals = None, old_model = None, batched_inputs: List[Dict[str, torch.Tensor]] = None, targets: Optional[List[Instances]] = None, old_targets: Optional[List[Instances]] = None) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training and old_proposals is not None:
            proposals = self.label_and_sample_proposals(origin_proposals, targets)
            old_proposals = self.old_label_and_sample_proposals(old_proposals, old_targets)
            om_proposals = self.old_label_and_sample_proposals(om_proposals, old_targets)
            
            losses = self.KD_forward_box(features, proposals, targets, old_features, old_proposals, old_targets, om_proposals, old_model)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self.KD_forward_mask(features, proposals, old_features, old_proposals, batched_inputs, images, om_proposals, old_model))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self.KD_forward_box(features, origin_proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
    
    def old_label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = self.old_add_ground_truth_to_proposals(targets, proposals)
            print("Proposal_append_gt")

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.pred_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.pred_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.pred_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("pred_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
        
    def old_add_ground_truth_to_proposals(self, 
        gt: Union[List[Instances], List[Boxes]], proposals: List[Instances]
    ) -> List[Instances]:
        """
        Call `add_ground_truth_to_proposals_single_image` for all images.
        Args:
            gt(Union[List[Instances], List[Boxes]): list of N elements. Element i is a Instances
                representing the ground-truth for image i.
            proposals (list[Instances]): list of N elements. Element i is a Instances
                representing the proposals for image i.
        Returns:
            list[Instances]: list of N Instances. Each is the proposals for the image,
                with field "proposal_boxes" and "objectness_logits".
        """
        assert gt is not None

        if len(proposals) != len(gt):
            raise ValueError("proposals and gt should have the same length as the number of images!")
        if len(proposals) == 0:
            return proposals

        return [
            self.old_add_ground_truth_to_proposals_single_image(gt_i, proposals_i)
            for gt_i, proposals_i in zip(gt, proposals)
        ]
        
        
    def old_add_ground_truth_to_proposals_single_image(self, 
        gt: Union[Instances, Boxes], proposals: Instances
    ) -> Instances:
        """
        Augment `proposals` with `gt`.
        Args:
            Same as `add_ground_truth_to_proposals`, but with gt and proposals
            per image.
        Returns:
            Same as `add_ground_truth_to_proposals`, but for only one image.
        """
        if isinstance(gt, Boxes):
            # convert Boxes to Instances
            gt = Instances(proposals.image_size, gt_boxes=gt)
        
        gt_boxes = gt.pred_boxes
        device = proposals.objectness_logits.device
        # Assign all ground-truth boxes an objectness logit corresponding to
        # P(object) = sigmoid(logit) =~ 1.
        gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
        gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

        # Concatenating gt_boxes with proposals requires them to have the same fields
        gt_proposal = Instances(proposals.image_size, **gt.get_fields())
        gt_proposal.proposal_boxes = gt_boxes
        gt_proposal.objectness_logits = gt_logits

        for key in proposals.get_fields().keys():
            assert gt_proposal.has(
                key
            ), "The attribute '{}' in `proposals` does not exist in `gt`".format(key)

        # NOTE: Instances.cat only use fields from the first item. Extra fields in latter items
        # will be thrown away.
        new_proposals = Instances.cat([proposals, gt_proposal])

        return new_proposals

    
    def KD_forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], old_features: Dict[str, torch.Tensor] = None, old_instances: List[Instances] = None, batched_inputs = None, images = None, om_proposals = None, old_model = None):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances
            return {} if self.training else old_instances

        if self.training and old_instances is not None:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            old_instances, _ = self.old_select_foreground_proposals(old_instances, self.num_classes)
            o_instances, _ = self.old_select_foreground_proposals(om_proposals, self.num_classes)
            
            new_o_instances = []
            idxs = []
            idx = -1
            
            for x, y in zip(old_instances, o_instances):
                temp_instances = Instances.cat([x, y])
            new_o_instances.append(temp_instances)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            old_features = [old_features[f] for f in old_model.model.roi_heads.mask_in_features]
            boxes = [x.proposal_boxes if self.training and old_instances is not None else x.pred_boxes for x in instances]
            old_boxes = [x.proposal_boxes if self.training and old_instances is not None else x.pred_boxes for x in new_o_instances] 
            o_boxes = [x.proposal_boxes if self.training and old_instances is not None else x.pred_boxes for x in new_o_instances]
            
            c_features = self.mask_pooler(features, boxes)
            o_features = self.mask_pooler(features, old_boxes)
            features = torch.cat((c_features, o_features), 0)
            old_features = old_model.model.roi_heads.mask_pooler(old_features, o_boxes)
            
            old_features = old_model.model.roi_heads.mask_head.layers(old_features)
        else:
            features = {f: features[f] for f in self.mask_in_features}
            old_features = {f: old_features[f] for f in self.mask_in_features}
        
        return self.mask_head(features, instances, new_o_instances, batched_inputs, images, old_features, new_o_instances)
        
    def old_select_foreground_proposals(self,
        proposals: List[Instances], bg_label: int
    ) -> Tuple[List[Instances], List[torch.Tensor]]:
        """
        Given a list of N Instances (for N images), each containing a `gt_classes` field,
        return a list of Instances that contain only instances with `gt_classes != -1 &&
        gt_classes != bg_label`.
        Args:
            proposals (list[Instances]): A list of N Instances, where N is the number of
                images in the batch.
            bg_label: label index of background class.
        Returns:
            list[Instances]: N Instances, each contains only the selected foreground instances.
            list[Tensor]: N boolean vector, correspond to the selection mask of
                each Instances object. True for selected instances.
        """
        assert isinstance(proposals, (list, tuple))
        assert isinstance(proposals[0], Instances)
        assert proposals[0].has("pred_classes")
        fg_proposals = []
        fg_selection_masks = []
        for proposals_per_image in proposals:
            gt_classes = proposals_per_image.pred_classes
            fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
            fg_idxs = fg_selection_mask.nonzero().squeeze(1)
            fg_proposals.append(proposals_per_image[fg_idxs])
            fg_selection_masks.append(fg_selection_mask)
        return fg_proposals, fg_selection_masks
    
    
    def KD_forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets: Optional[List[Instances]] = None, old_features: Dict[str, torch.Tensor] = None, old_proposals: List[Instances] = None, old_targets: Optional[List[Instances]] = None, o_proposals = None, old_model = None):  
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        if self.training and old_features is not None:
            
            o_features = [old_features[f] for f in self.box_in_features]
            head_outputs = []  
            new_o_proposals = []
            prev_pred_boxes = None
            old_prev_pred_boxes = None 
            o_prev_pred_boxes = None 
            old_image_sizes = [x.image_size for x in old_proposals]
            o_image_sizes = [x.image_size for x in o_proposals]
            
            for x, y in zip(old_proposals, o_proposals):
                temp_propose = Instances.cat([x, y])
            new_o_proposals.append(temp_propose)
            
            old_proposals = new_o_proposals
            o_proposals = new_o_proposals
            fea_loss = 0
            
            for k in range(self.num_cascade_stages):
                if k > 0:
                    # The output boxes of the previous stage are used to create the input
                    # proposals of the next stage.
                    proposals = self.incre_create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                    old_proposals = self.incre_create_proposals_from_boxes(old_prev_pred_boxes, old_image_sizes)
                    o_proposals = old_model.model.roi_heads._create_proposals_from_boxes(o_prev_pred_boxes, o_image_sizes)
                    
                    proposals = self._match_and_label_boxes(proposals, k, targets)
                    old_proposals = self.old_match_and_label_boxes(old_proposals, k, old_targets) 
                    o_proposals = self.old_match_and_label_boxes(o_proposals, k, old_targets)
                    
                predictions = self._run_stage(features, proposals, k)
                old_predictions = self._run_stage(features, old_proposals, k)
                o_predictions = old_model.model.roi_heads._run_stage(o_features, o_proposals, k)
                
                if self.feaKD:
                    old_fea = self.loss_run_stage(features, old_proposals, k)
                    o_fea = self.loss_run_stage(o_features, old_proposals, k)
                    for (old_feamap, o_feamap) in zip(old_fea, o_fea) :
                        a = old_feamap
                        b = o_feamap
            
                        if isinstance(a, list):
                            layer_loss = torch.tensor(
                                [torch.norm(aa - bb, p=2, dim=-1) for aa, bb in zip(a, b)]
                            ).to(self.device)
                        else:
                            layer_loss = torch.norm(a - b, p=2, dim=-1)
                        fea_loss = fea_loss + torch.mean(layer_loss)
                
                prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                old_prev_pred_boxes = self.box_predictor[k].predict_boxes(old_predictions, old_proposals)
                o_prev_pred_boxes = old_model.model.roi_heads.box_predictor[k].predict_boxes(o_predictions, o_proposals)
                head_outputs.append((self.box_predictor[k], predictions, proposals, old_predictions, old_proposals, o_predictions, o_proposals))
            
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals, old_predictions, old_proposals, o_predictions, o_proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.KD_losses(predictions, proposals, old_predictions, old_proposals, o_predictions, o_proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            if self.feaKD:
                features_loss = {
                    "features_loss": 0.01*fea_loss/3,  
                }
                losses.update(features_loss)
            
            return losses

        else:
            for k in range(self.num_cascade_stages):
                if k > 0:
                    # The output boxes of the previous stage are used to create the input
                    # proposals of the next stage.
                    proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                predictions = self._run_stage(features, proposals, k)
                prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                head_outputs.append((self.box_predictor[k], predictions, proposals))
            
            
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances
    

    def incre_create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)
        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
            
    
    @torch.no_grad()
    def old_match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.
        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances
        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.pred_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.pred_classes[matched_idxs]
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.pred_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.pred_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.pred_classes = gt_classes
            proposals_per_image.pred_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
cfg.DATASETS.TRAIN = ("SegPC_train",)
cfg.DATASETS.TEST = ("SegPC_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = os.path.join(args.weight_folder, args.weight_file)  
cfg.SOLVER.IMS_PER_BATCH = args.batch_size
cfg.SOLVER.BASE_LR = 0.02/8
cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'

cfg.SOLVER.WARMUP_ITERS = 1500 
cfg.SOLVER.MAX_ITER = args.iterations 
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
if args.num_class:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class

cfg.TEST.EVAL_PERIOD = 1000 

if 'Original' not in args.backbone:
  cfg.MODEL.BACKBONE.NAME = args.backbone
  
cfg.CUDNN_BENCHMARK = True
cfg.OUTPUT_DIR = args.work_dir

cfg.MODEL.META_ARCHITECTURE = "KD_CRCNN"
cfg.MODEL.ROI_HEADS.NAME = "KD_CROIHeads"

# Incremental learning components
cfg.outKD = args.outKD
cfg.feaKD = args.feaKD
cfg.PseudoLabel = args.PseudoLabel



from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()



  

  


