from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, softmax_helper
from nnunet.training.loss_functions.focal_loss import FocalLoss
from nnunet.losses_pytorch.hausdorff import HausdorffDTLoss, HausdorffERLoss
from nnunet.losses_pytorch.dice_loss import GDiceLoss
import torch.nn.functional as F
import torch.nn as nn
import torch

class GDL_and_CE_loss(DC_and_CE_loss):
    def __init__(self, dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                 log_dice, ignore_label)

        self.dc = GDiceLoss(apply_nonlin=softmax_helper, **dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DL_and_WCE_loss(DC_and_CE_loss):
    def __init__(self, dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                 log_dice, ignore_label)

        self.ce = WeightedCrossEntropyLoss()

class DL_and_Focal_loss(DC_and_CE_loss):
    def __init__(self, dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                 log_dice, ignore_label)

        self.ce = FocalLoss(apply_nonlin=softmax_helper, **ce_kwargs)


class GDL_and_WCE_loss(GDL_and_CE_loss):
    def __init__(self, dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                 log_dice, ignore_label)

        self.ce = WeightedCrossEntropyLoss()


class Hau_and_CE_loss(GDL_and_CE_loss):
    def __init__(self, dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                 log_dice, ignore_label)

        self.dc = HausdorffERLoss()


class WeightedCrossEntropyLoss(nn.Module):

    def forward(self, net_output, gt):

        if len(net_output.shape) == len(gt.shape):
            assert gt.shape[1] == 1
            gt = gt[:, 0]

        with torch.no_grad():
            element, count = torch.unique(gt, return_counts=True)
            # count = count ** 2
            weight_exist = count.sum() / count

            weight = torch.ones(net_output.shape[1]).cuda()
            weight = weight.scatter_(0, element, weight_exist)
            # print(weight)

        return F.cross_entropy(net_output, gt, weight=weight)



