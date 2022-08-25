import argparse

args = argparse.ArgumentParser()
args.add_argument('--backbone',type=str,required=True,choices=["Original","Effb5","Transformer_Effb5"],help="The backbone to be used from the given choices")
args.add_argument('--saved_model_path',type=str,required=True,help="path to the saved model which will be loaded")
args.add_argument('--input_images_folder',type=str,required=True,help="path to the folder where images to inference on are kept")
args.add_argument('--save_path',type=str,required=True,help="path to the folder where the generated masks will be saved")

args = args.parse_args()

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import math
import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model,build_resnet_backbone,build_backbone
from detectron2.structures import ImageList
from detectron2.structures import Instances
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



from detectron2.data.datasets import register_coco_instances



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
class KD_RCNN(GeneralizedRCNN):
        
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
        print('old_model_checkpoint_start')
        ocfg = get_cfg()
        ocfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        ocfg.MODEL.WEIGHTS = os.path.join("/home/frank/Desktop/instance segmentation/SegPC-2021-main/model_and_log/model_and_log_nuclei/", "model_final.pth")  # path to the model we just trained
        ocfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        ocfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        old_model = DefaultPredictor(ocfg)
        
        images = self.preprocess_image(batched_inputs)
        #old_instances = old_model.model.inference(batched_inputs)
        old_features = old_model.model.backbone(images.tensor) #將新圖片輸入舊模型以做取得 old_features
        old_proposals,_ = old_model.model.proposal_generator(images, old_features, None)  #將新圖片輸入舊模型以做取得 old_features
        old_instances = old_model.model.roi_heads._forward_box(old_features, old_proposals)
        old_instances = old_model.model.roi_heads._forward_mask(old_features, old_instances)
        old_instances = old_model.model.roi_heads._forward_keypoint(old_features, old_instances)
        print('old_model_checkpoint_end')
        #print("old_features:\n", old_features)
        #print("old_proposals:\n", old_proposals)
        
        
        #print("imrcnn_old_instances:")
        #print(old_instances)
        #print("====================")
        #old_instances = old_model.model.roi_heads.KD_forward_box(old_features, old_proposals)
        #old_instances, _ = old_model.model.roi_heads(images, old_features, old_proposals, None)

        
        if not self.training:
            print('in inference')
            return self.inference(batched_inputs)
        print('not in inference')
        images = self.preprocess_image(batched_inputs)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            
        #print("gt_instances:")
        #print(gt_instances)    #檢查哪裡有寫產生gt_masks, gt_box
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

        _, detector_losses = self.roi_heads(images, features, proposals, old_features, old_proposals, old_model, gt_instances, old_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

@ROI_HEADS_REGISTRY.register()
class KD_ROIHeads(StandardROIHeads):
    def forward( self, images: ImageList, features: Dict[str, torch.Tensor], origin_proposals: List[Instances], old_features: Dict[str, torch.Tensor] = None, old_proposals: List[Instances] = None, old_model = None, targets: Optional[List[Instances]] = None, old_targets: Optional[List[Instances]] = None) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        
        del images
        #if self.training:
            #assert targets, "'targets' argument is required during training"
            
        

        if self.training and old_proposals is not None:
        
            proposals = self.label_and_sample_proposals(origin_proposals, targets)
            del targets
            #produce old_proposals
            #old_instances = self._forward_box(old_features, old_proposals)#==============================================
            #old_instances = self.forward_with_given_boxes(old_features, old_instances)
            
            #old_proposals = self.old_label_and_sample_proposals(old_proposals, old_targets)
            old_proposals = self.old_label_and_sample_proposals(origin_proposals, old_targets)
            
            #print('proposals')
            #print(proposals)
            #print('old_proposals')
            #print(old_proposals)
            
            losses = self.KD_forward_box(features, proposals, old_features, old_proposals, old_model)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self.KD_forward_mask(features, proposals, old_proposals))
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

    
    def KD_forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], old_instances: List[Instances] = None):
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
        #print("pred_mask_logits_start")
        #print(old_instances)
        #print(features)
        if not self.mask_on:
            return {} if self.training else instances
            return {} if self.training else old_instances

        if self.training and old_instances is not None:
            # head is only trained on positive proposals.
            ##print(self.num_classes)
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            old_instances, _ = self.old_select_foreground_proposals(old_instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training and old_instances is not None else x.pred_boxes for x in instances]
            old_boxes = [x.proposal_boxes if self.training and old_instances is not None else x.pred_boxes for x in old_instances] #會出現 tensor size錯誤在這裏
            
            #print("boxes", type(boxes))
            #print(boxes)
            cur_features = self.mask_pooler(features, boxes)
            old_features = self.mask_pooler(features, old_boxes)
            features = torch.cat((cur_features, old_features), 0)
            #print('cur_features')
            #print(cur_features)
            #print('old_features')
            #print(old_features)
            #print("mask_pooler")
            #print(features.size())
        else:
            features = {f: features[f] for f in self.mask_in_features}
        #print("pred_mask_logits")
        #print(features)
        return self.mask_head(features, instances, old_instances)
        
    #related to KD_forward_mask
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
    
    
    def KD_forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], old_features: Dict[str, torch.Tensor] = None, old_proposals: List[Instances] = None, old_model = None):
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
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training and old_features is not None:
            
            #old_features = [old_features[f] for f in self.box_in_features]
            #old_box_features = old_model.model.roi_heads.box_pooler(old_features, [x.proposal_boxes for x in old_proposals])
            #old_box_features = old_model.model.roi_heads.box_head(old_box_features)
            #old_predictions = old_model.model.roi_heads.box_predictor(old_box_features)
            #del old_box_features
            
            #features = [features[f] for f in self.box_in_features]
            old_box_features = self.box_pooler(features, [x.proposal_boxes for x in old_proposals])
            old_box_features = self.box_head(old_box_features)
            old_predictions = self.box_predictor(old_box_features)
            del old_box_features
            
            #losses = self.box_predictor.losses(predictions, proposals)
            losses = self.box_predictor.KD_losses(predictions, proposals, old_predictions, old_proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
            
    def KD_losses_noused(self, predictions, proposals, old_predictions):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        #old model predicted classes
        old_classes = (
            cat([p.pred_classes for p in old_predictions], dim=0) if len(old_predictions) else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
            old_boxes = cat(
                [(p.pred_boxes if p.has("pred_boxes") else p.proposal_boxes).tensor for p in old_predictions],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean") + cross_entropy(scores, old_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ) + self.box_reg_loss(proposal_boxes, old_boxes, proposal_deltas, old_classes),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        
        
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

if 'Original' not in args.backbone:
  cfg.MODEL.BACKBONE.NAME = args.backbone

cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2


cfg.MODEL.WEIGHTS = args.saved_model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
predictor = DefaultPredictor(cfg)

img_root = args.input_images_folder
pred_root = args.save_path
names = os.listdir(img_root)
thresh = 0.0
var= 1

cfg.MODEL.META_ARCHITECTURE = "KD_RCNN"
cfg.MODEL.ROI_HEADS.NAME = "KD_ROIHeads"

res_size=(1080,1440)

for name in names:
    print(var)
    var+=1
    print(name)
    index = name[:-4]
    
    im = cv2.imread(img_root+name)
    orig_shape = im.shape[0:2]
    im = cv2.resize(im, res_size[::-1],interpolation=cv2.INTER_NEAREST)

    outputs = predictor(im)
    scores = outputs['instances'].to('cpu').scores.numpy()
    pred_masks = outputs['instances'].to('cpu').pred_masks.numpy()
    pred_classes = outputs['instances'].to('cpu').pred_classes.numpy()
    print('pred_classes')
    print(pred_classes)
    count = 1
    for i in range(len(scores)):
        
        if scores[i]>=thresh:
           tmp_mask = pred_masks[i].astype('uint8')
           if (pred_classes[i] == 0):
               tmp_mask = 255*tmp_mask
               print('nu')
           elif (pred_classes[i] == 1):
               tmp_mask = 255*tmp_mask
               print('cy')
           else:
               continue
           tmp_mask = cv2.resize(tmp_mask, orig_shape[::-1],interpolation=cv2.INTER_NEAREST)
           cv2.imwrite(pred_root+index+'_'+str(count)+'.bmp', tmp_mask)
           count+=1
    if count==1:
        tmp_mask = np.zeros(res_size)
        cv2.imwrite(pred_root+index+'_1.bmp', tmp_mask)
        print('blank mask saved')
    print(count-1,'masks saved')



  

  


