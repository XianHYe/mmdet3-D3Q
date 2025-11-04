# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple
from ..utils import multi_apply
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean, ConfigType
from ..layers import inverse_sigmoid
from .detr_head import DETRHead
from ..losses.smooth_l1_loss import SmoothL1Loss
from ..utils import gaussian_radius

@MODELS.register_module()
class DeformableDETRHead_D3Q(DETRHead):
    r"""Head of DeformDETR: Deformable DETR: Deformable Transformers for
    End-to-End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        share_pred_layer (bool): Whether to share parameters for all the
            prediction layers. Defaults to `False`.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
    """

    def __init__(self,
                 *args,
                 num_classes: int,
                 loss_density_map: ConfigType = dict (type='GaussianFocalLoss_T', loss_weight=1.0),
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
                 loss_d_cls=dict(
                    type='CrossEntropyLoss',
                    bg_cls_weight=0.1,
                    use_sigmoid=True,
                    loss_weight=1.0,
                    class_weight=1.0),
                 train_cfg: ConfigType = dict(
                    assigner_d=dict(
                        type='HungarianAssigner_cpu',
                        match_costs=[
                            dict(type='ClassificationCost', weight=1.),
                            dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                            dict(type='IoUCost', iou_mode='giou', weight=2.0)
                        ])),
                 **kwargs) -> None:
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage  

        self.d_use_sigmoid = loss_d_cls.get('use_sigmoid', None)
        if self.d_use_sigmoid:
            self.d_cls_out_channels = num_classes
        else:
            self.d_cls_out_channels = num_classes + 1       

        super().__init__(*args, num_classes=num_classes, train_cfg=train_cfg, **kwargs)
        self.densitymap_head = self._build_head(self.embed_dims, self.embed_dims, 1)
        if train_cfg:
            assert 'assigner_d' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner_d = train_cfg['assigner_d']
            self.assigner_d = TASK_UTILS.build(assigner_d)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
 

        class_weight = loss_d_cls.get('class_weight', None)
       
        d_bg_cls_weight = loss_d_cls.get('bg_cls_weight', class_weight)
        if self.d_use_sigmoid:
            class_weight = torch.ones(num_classes)
        else:
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = d_bg_cls_weight
            loss_d_cls.update({'class_weight': class_weight})
        if 'bg_cls_weight' in loss_d_cls:
            loss_d_cls.pop('bg_cls_weight')
        self.d_bg_cls_weight = d_bg_cls_weight   

    
        self.density_map_loss = MODELS.build(loss_density_map)
        self.loss_d_cls = MODELS.build(loss_d_cls)


    def _build_head(self, in_channels: int, feat_channels: int,
                    out_channels: int) -> nn.Sequential:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1))
        # layers = nn.ModuleList( 
        #     [copy.deepcopy(layer) for _ in range(self.num_layers)]
        #     )
        return layer

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_d_cls = Linear(self.embed_dims, self.d_cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = [fc_d_cls for _ in range(self.num_pred_layer -1)]
            self.cls_branches.append(fc_cls)
            self.cls_branches =  nn.ModuleList(self.cls_branches)
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = [copy.deepcopy(fc_d_cls) for _ in range(self.num_pred_layer -1)]
            self.cls_branches.append(copy.deepcopy(fc_cls))
            self.cls_branches =  nn.ModuleList(self.cls_branches)
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
                
        bias_init = bias_init_with_prob(0.1)
        # for m in self.densitymap_head:
        self.densitymap_head[-1].bias.data.fill_(bias_init)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def density_map_forward(self, memory: Tensor, spatial_shapes: Tensor):
        bs, _, d = memory.shape
        H, W = spatial_shapes[0]
        x = memory[:, :H*W, :].contiguous().view(bs, H, W, d).permute(0, 3, 1, 2)
        densitymap_pred = self.densitymap_head(x).sigmoid()
        return densitymap_pred
    
    def loss_densitymap_head(self, densitymap_pred: Tensor, batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],):
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']
        densitymap_target, avg_factor = self.get_targets_densitymap_head(gt_bboxes, gt_labels,
                                                     densitymap_pred.shape,
                                                     img_shape)
        loss_densitymap = self.density_map_loss(densitymap_pred, densitymap_target, avg_factor=avg_factor)

        return loss_densitymap

    def get_targets_densitymap_head(self, gt_bboxes: List[Tensor], gt_labels: List[Tensor],
                    feat_shape: tuple, img_shape: tuple) -> Tuple[dict, int]:
        img_h, img_w = img_shape[:2]
        bs, c, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        densitymap_target = gt_bboxes[-1].new_zeros(
            [bs, c, feat_h, feat_w])
        
        avg_factor = None
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            
            if avg_factor is None:
                avg_factor = gt_label.size(0)
            else:
                avg_factor += gt_label.size(0)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(1, int(radius))
                self.gen_gaussian_target(densitymap_target[batch_id, 0],
                                    [ctx_int, cty_int], radius)
        
        return densitymap_target, avg_factor
                
    def gaussian2D(self, radius, sigma=1, dtype=torch.float32, device='cpu'):
        """Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma (int): Sigma of gaussian function. Default: 1.
            dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
            device (str): Device of gaussian tensor. Default: 'cpu'.

        Returns:
            h (Tensor): Gaussian kernel with a
                ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        x = torch.arange(
            -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
        y = torch.arange(
            -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

        h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        h = h/h.sum()
        return h


    def gen_gaussian_target(self, heatmap, center, radius, k=1):
        """Generate 2D gaussian heatmap.

        Args:
            heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
                it and maintain the max value.
            center (list[int]): Coord of gaussian kernel's center.
            radius (int): Radius of gaussian kernel.
            k (int): Coefficient of gaussian kernel. Default: 1.

        Returns:
            out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
        """
        diameter = 2 * radius + 1
        gaussian_kernel = self.gaussian2D(
            radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

        x, y = center

        height, width = heatmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                        radius - left:radius + right]
        out_heatmap = heatmap
        torch.max(
            masked_heatmap,
            masked_gaussian * k,
            out=out_heatmap[y - top:y + bottom, x - left:x + right])

        return out_heatmap


    def loss(self, hidden_states: List[Tensor], references: List,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             densitymap_pred: Tensor, query_num: Tensor, batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passed in,
                otherwise it would be `None`.
            enc_outputs_coord (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # outs = self(hidden_states, references)
        outs = multi_apply(
            self,
            hidden_states,
            references
        )

        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs, densitymap_pred, query_num)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: List[Tensor],
        all_layers_bbox_preds: List[Tensor],
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        densitymap_pred: Tensor,
        query_num: Tensor,
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passes in,
                otherwise, it would be `None`.
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = self.loss_by_feat_d(all_layers_cls_scores,
                                         all_layers_bbox_preds,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)

        # query_num_tgt = torch.tensor([batch_gt.labels.numel() for batch_gt in batch_gt_instances], device=query_num.device)
        # density_loss = 0.3* self.query_num_loss(query_num, query_num_tgt)

        loss_densitymap = self.loss_densitymap_head(densitymap_pred, batch_gt_instances, batch_img_metas)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            for i in range(len(proposal_gt_instances)):
                proposal_gt_instances[i].labels = torch.zeros_like(
                    proposal_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
            # loss_dict['density_loss'] = density_loss
            loss_dict['loss_densitymap'] = loss_densitymap
        return loss_dict

    def loss_by_feat_d(
        self,
        all_layers_cls_scores: List[Tensor],
        all_layers_bbox_preds: List[Tensor],
        # mask_k: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'
        losses_cls_all=[]
        losses_bbox_all=[]
        losses_iou_all=[]
        batch_size = len(all_layers_cls_scores)
        for i in range(batch_size):
            losses_cls_i, losses_bbox_i, losses_iou_i = multi_apply(
                self.loss_by_feat_single_d,
                all_layers_cls_scores[i],
                all_layers_bbox_preds[i],
                batch_gt_instances=batch_gt_instances[i],
                batch_img_metas=batch_img_metas[i])
            # if losses_cls is None:
            #     losses_cls = torch.tensor(losses_cls_i)
            # else:
            #     losses_cls += losses_cls_i
            
            # if losses_bbox is None:
            #     losses_bbox = torch.tensor(losses_bbox_i)
            # else:
            #     losses_bbox += losses_bbox_i

            # if losses_iou is None:
            #     losses_iou = torch.tensor(losses_iou_i)
            # else:
            #     losses_iou += losses_iou_i
            losses_cls_all.append(losses_cls_i)
            losses_bbox_all.append(losses_bbox_i)
            losses_iou_all.append(losses_iou_i)

        # losses_cls = torch.tensor(losses_cls)
        # losses_bbox = torch.tensor(losses_bbox)
        # losses_iou = torch.tensor(losses_iou)

        # losses_cls = torch.sum(losses_cls, dim=0)/batch_size
        # losses_bbox = torch.sum(losses_bbox, dim=0)/batch_size
        # losses_iou = torch.sum(losses_iou, dim=0)/batch_size

        # losses_cls = losses_cls/batch_size
        # losses_bbox = losses_bbox/batch_size
        # losses_iou = losses_iou/batch_size

        loss_dict = dict()
        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls_all[-1]
        # loss_dict['loss_bbox'] = losses_bbox_all[-1]
        # loss_dict['loss_iou'] = losses_iou_all[-1]
        # loss from other decoder layers
        # num_dec_layer = 0
        batch_id = 0
        for losses_cls, losses_bbox, losses_iou in \
                    zip(losses_cls_all, losses_bbox_all, losses_iou_all):
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in \
                    zip(losses_cls[:], losses_bbox[:], losses_iou[:]):
                if batch_id==0:
                    if num_dec_layer!=5:
                        loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i/batch_size
                        loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i/batch_size
                        loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i/batch_size
                    else:
                        # loss from the last decoder layer
                        loss_dict['loss_cls'] = loss_cls_i/batch_size
                        loss_dict['loss_bbox'] = loss_bbox_i/batch_size
                        loss_dict['loss_iou'] = loss_iou_i/batch_size
                                     
                else:
                    if num_dec_layer!=5:
                        loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i/batch_size
                        loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i/batch_size
                        loss_dict[f'd{num_dec_layer}.loss_iou'] += loss_iou_i/batch_size

                    else:
                        loss_dict['loss_cls'] += loss_cls_i/batch_size
                        loss_dict['loss_bbox'] += loss_bbox_i/batch_size
                        loss_dict['loss_iou'] += loss_iou_i/batch_size
                
                num_dec_layer += 1
            batch_id += 1
        return loss_dict

    def loss_by_feat_single_d(self, cls_scores: Tensor, bbox_preds: Tensor,
                            # mask_k: Tensor,
                            batch_gt_instances: InstanceData,
                            batch_img_metas: dict) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        batch_gt_instances = [batch_gt_instances]
        batch_img_metas = [batch_img_metas]
        cls_reg_targets = self.get_targets_d(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.d_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.d_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_d_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
    
    def get_targets_d(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single_d,
                                      cls_scores_list, bbox_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single_d(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner_d.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)


    def predict(self,
                hidden_states: List[Tensor],
                references: List,
                batch_data_samples: SampleList,
                query_num: Tensor,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # outs = self(hidden_states, references)
        outs = multi_apply(
            self,
            hidden_states,
            references
        )

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas,  rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: List[Tensor],
                        all_layers_bbox_preds: List[Tensor],
                        batch_img_metas: List[Dict],
                        # mask_k: Tensor,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        assert len(all_layers_cls_scores)==len(all_layers_bbox_preds)==1
        cls_scores = all_layers_cls_scores[0][-1]
        bbox_preds = all_layers_bbox_preds[0][-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            # mask_k_i = mask_k[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                    # mask_k_i,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                # mask_k: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        # max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        # max_per_img = mask_k.item()
        max_per_img = cls_score.shape[0]
        img_shape = img_meta['img_shape']
        # exclude background
        if self.d_use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

