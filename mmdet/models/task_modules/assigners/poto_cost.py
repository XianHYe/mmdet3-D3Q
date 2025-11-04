# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Optional, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps, bbox_xyxy_to_cxcywh
from .match_cost import BaseMatchCost

@TASK_UTILS.register_module()
class ClassificationCost_sigmoid(BaseMatchCost):
    """ClsSoftmaxCost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ...  match_costs.match_cost import ClassificationCost
        >>> import torch
        >>> self = ClassificationCost()
        >>> cls_pred = torch.rand(4, 3)
        >>> gt_labels = torch.tensor([0, 1, 2])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(cls_pred, gt_labels)
        tensor([[-0.3430, -0.3525, -0.3045],
            [-0.3077, -0.2931, -0.3992],
            [-0.3664, -0.3455, -0.2881],
            [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        pred_scores = pred_scores.sigmoid()
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight

class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


@TASK_UTILS.register_module()
class POTOCost(BaseMatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 iou_mode: str = 'giou',
                 poto_alpha = 0.8,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format
        self.iou_mode = iou_mode
        self.poto_alpha = poto_alpha

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pred_scores = pred_instances.scores
        pred_bboxes = pred_instances.bboxes


        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        if self.iou_mode == 'giou':
            iou_cost = (overlaps+1)/2
        else:
            iou_cost = overlaps

        pred_probs = pred_scores.sigmoid()
        pred_probs = pred_probs[:, gt_labels]
        potocost = pred_probs**(1-self.poto_alpha)* iou_cost**self.poto_alpha
        
        return -potocost * self.weight


@TASK_UTILS.register_module()
class TinyBBoxL1Cost(BaseMatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # convert box format
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            pred_bboxes = bbox_xyxy_to_cxcywh(pred_bboxes)

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        gt_bboxes_w = gt_bboxes[..., 2]
        pred_bboxes_w = pred_bboxes[..., 2]
        gt_bboxes_h= gt_bboxes[..., 3]
        pred_bboxes_h = pred_bboxes[..., 3]
        # gt_bboxes_wh = gt_bboxes[..., 2:]
        # pred_bboxes_wh = pred_bboxes[..., 2:]


        bbox_cost_w = torch.abs(torch.log(pred_bboxes_w[:, None]/gt_bboxes_w[None, :]))
        bbox_cost_h = torch.abs(torch.log(pred_bboxes_h[:, None]/gt_bboxes_h[None, :]))
        # bbox_cost_wh = torch.abs(torch.log(pred_bboxes_wh[:, None, :]/gt_bboxes_wh[None, :, : ]))


        bbox_cost = torch.cdist(pred_bboxes[..., :2], gt_bboxes[..., :2], p=1)
        bbox_cost += bbox_cost_w
        bbox_cost += bbox_cost_h

        return bbox_cost * self.weight

