import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from typing import Optional
from mmdet.models.losses.utils import weighted_loss
from mmdet.registry import MODELS
from .utils import weight_reduce_loss

def quality_focal_loss_with_logic(pred, target, reduction='mean', avg_factor=None, beta=2.0):

    label, score, weight = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta) * 0.75

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()

    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * weight[pos]
    loss = loss.sum(-1)
    return loss

def quality_focal_loss_f_with_logic(pred, target: tuple, weight, reduction='mean', avg_factor=None, beta=2.0):

    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta) * 0.75

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()

    pos_weight = abs(score[pos] - pred[pos, pos_label])
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * pos_weight.pow(beta)
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    # loss = loss.sum(-1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class SotfFocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super(SotfFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in SFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if not self.activated:
                calculate_loss_func = quality_focal_loss_with_logic
            else:
                raise NotImplementedError
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                beta=self.beta,
                )
        else:
            raise NotImplementedError
        return loss_cls

class SotfFocalLoss_f(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super(SotfFocalLoss_f, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in SFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if not self.activated:
                # if pred.dim() == target.dim():
                #     calculate_loss_func = quality_focal_loss_f_with_logic
                # else:
                #     num_classes = pred.size(1)
                #     target = F.one_hot(target, num_classes=num_classes + 1)
                #     target = target[:, :num_classes]
                #     calculate_loss_func = quality_focal_loss_f_with_logic
                
                calculate_loss_func = quality_focal_loss_f_with_logic
            else:
                raise NotImplementedError
            
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                beta=self.beta,
                )
        else:
            raise NotImplementedError
        return loss_cls
    

@weighted_loss
def tinyl1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    
    eps = torch.finfo(torch.float32).eps
    # pred[..., 2:] = torch.log(pred[..., 2:] / pred[..., 2:])
    # target[..., 2:] = torch.log(target[..., 2:] / pred[..., 2:])

    # loss_wh = torch.abs(torch.log(pred[:, :, None, 2:] / target[:, None, :,  2:]))
    # loss_h = torch.abs(torch.log(pred[..., :, None, 3] / target[..., None, :,  3]))

    assert pred.size() == target.size()
    loss = torch.abs(pred[..., :2] - target[..., :2])

    loss_wh = torch.abs(torch.log(pred[..., 2:] / (target[..., 2:] + eps)))
    loss = torch.cat([loss, loss_wh], dim=-1)
    return loss

@MODELS.register_module()
class TinyL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * tinyl1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox