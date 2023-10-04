import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
import os
import cv2
import torch
import torch.nn as nn
import numpy as np


class RefineLoss(nn.Module):
    def __init__(self, alpha=1.5, alpha1=0.5, reduction="mean"):
        super(RefineLoss, self).__init__()
        self.alpha = alpha
        self.alpha1 = alpha1
        self.reduction = reduction
        self.fx = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.fy = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.lap = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # sobel filter
        ngx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        ngy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        # laplace filter
        f_lpa = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype= np.float32)

        self.fx.weight.data.copy_(torch.from_numpy(ngx))
        self.fy.weight.data.copy_(torch.from_numpy(ngy))
        self.lap.weight.data.copy_(torch.from_numpy(f_lpa))

        for param in self.fx.parameters():
            param.requires_grad = False
        for param in self.fy.parameters():
            param.requires_grad = False
        for param in self.lap.parameters():
            param.requires_grad = False


    def forward(self, grayimg, pred, mask):
        '''
        grayimg: gray scale input image
        pred: predicted mask
        mask: boundary mask. can be generate from ground truth foreground mask by  morphological transformation
        '''
        gx = self.fx(grayimg)
        gy = self.fy(grayimg)

        px = self.fx(pred)
        py = self.fy(pred)

        gm = torch.sqrt(gx * gx + gy * gy + 1e-6)
        pm = torch.sqrt(px * px + py * py + 1e-6)

        gv = (gx / gm, gy / gm)
        pv = (px / pm, py / pm)

        Lcos = (1 - torch.abs(gv[0] * pv[0] + gv[1] * pv[1])) * pm
        Lmag = torch.clamp_min(self.alpha * gm - pm, 0)

        Lrefine = (self.alpha1 * Lcos + (1 - self.alpha1) * Lmag) * mask

        if self.reduction == "mean":
            Lrefine = Lrefine.mean()
        elif self.reduction == "sum":
            Lrefine = Lrefine.sum()

        return Lrefine

class generate_edge(nn.Module):
    def __init__(self, alpha=1.5, alpha1=0.5, reduction="mean"):
        super(RefineLoss, self).__init__()
        self.alpha = alpha
        self.alpha1 = alpha1
        self.reduction = reduction
        self.fx = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.fy = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.l1_loss = nn.L1Loss()

        ngx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        ngy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        self.fx.weight.data.copy_(torch.from_numpy(ngx))
        self.fy.weight.data.copy_(torch.from_numpy(ngy))

        for param in self.fx.parameters():
            param.requires_grad = False
        for param in self.fy.parameters():
            param.requires_grad = False

    def forward(self, pred, edge):
        '''
        grayimg: gray scale input image
        pred: predicted mask
        mask: boundary mask. can be generate from ground truth foreground mask by  morphological transformation
        '''
        # gx = self.fx(grayimg)
        # gy = self.fy(grayimg)

        px = self.fx(pred)
        py = self.fy(pred)

        #gm = torch.sqrt(gx * gx + gy * gy + 1e-6)
        pm = torch.sqrt(px * px + py * py + 1e-6)
        #gv = (gx / gm, gy / gm)
        #pv = (px / pm, py / pm)

        #Lcos = (1 - torch.abs(gv[0] * pv[0] + gv[1] * pv[1])) * pm
        #Lmag = torch.clamp_min(self.alpha * gm - pm, 0)

        #Lrefine = (self.alpha1 * Lcos + (1 - self.alpha1) * Lmag) * mask

        # if self.reduction == "mean":
        #     Lrefine = Lrefine.mean()
        # elif self.reduction == "sum":
        #     Lrefine = Lrefine.sum()

        return pm

bce_loss = nn.BCELoss(reduction='mean')  # size_average
refine_loss = RefineLoss(reduction='mean')


def ba_loss(pred, target, ba, mask, grayimg):
    '''
    grayimg: gray scale input image
    pred: predicted mask
    mask: boundary mask. can be generate from ground truth foreground mask by  morphological transformation
    ba: predicted boundary attention
    '''
    alpha, beta, gamma = 0.6, 0.3, 0.1
    Lbound = bce_loss(ba, mask)
    Lseg = bce_loss(pred, target)
    Lrefine = refine_loss(grayimg, pred, mask)
    return alpha * Lseg + beta * Lbound + gamma * Lrefine


class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def parsing_loss_bk(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)

        target[1] = torch.clamp(target[1], 0, 1)
        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # loss for parsing
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])

        # loss for edge
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)

        return loss

    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)
        loss = 0

        # loss for parsing
        preds_parsing = preds
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target)
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target)

        return loss

    def forward(self, preds, target):
        loss = self.parsing_loss_bk(preds, target)
        return loss


class CriterionCrossEntropyEdgeParsing_boundary_attention_loss(nn.Module):
    """Weighted CE2P loss for face parsing.

    Put more focus on facial components like eyes, eyebrow, nose and mouth
    """

    def __init__(self, loss_weight=[1.0, 1.0, 1.0], ignore_index=255, num_classes=11):
        super(CriterionCrossEntropyEdgeParsing_boundary_attention_loss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion_weight = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        self.loss_weight = loss_weight

    def forward(self, preds, target): # preds: seg, edge,   targets: seg, edge
        h, w = target[0].size(1), target[0].size(2)

        input_labels = target[1].data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels == 1).astype(np.float)
        neg_num = np.sum(input_labels == 0).astype(np.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = (weight_neg, weight_pos)
        weights = torch.from_numpy(np.array(weights)).float().cuda()

        edge_p_num = target[1].cpu().numpy().reshape(target[1].size(0), -1).sum(axis=1)
        edge_p_num = np.tile(edge_p_num, [h, w, 1]).transpose(2, 1, 0)
        edge_p_num = torch.from_numpy(edge_p_num).cuda().float()

        loss_edge = 0
        loss_parse = 0
        loss_att_edge = 0

        for i in range(len(preds)):
            preds_i_ = preds[i]
            scale_parse = F.upsample(input=preds_i_[0], size=(h, w), mode='bilinear')  # parsing
            scale_edge = F.upsample(input=preds_i_[1], size=(h, w), mode='bilinear')  # edge

            loss_parse_ = self.criterion(scale_parse, target[0])
            loss_edge_ = F.cross_entropy(scale_edge, target[1], weights)
            loss_att_edge_ = self.criterion_weight(scale_parse, target[0]) * target[1].float()
            loss_att_edge_ = loss_att_edge_ / edge_p_num  # only compute the edge pixels
            loss_att_edge_ = torch.sum(loss_att_edge_) / target[1].size(0)  # mean for batchsize

            loss_parse += loss_parse_
            loss_edge += loss_edge_
            loss_att_edge += loss_att_edge_

        # print('loss_parse: {}\t loss_edge: {}\t loss_att_edge: {}'.format(loss_parse,loss_edge,loss_att_edge))
        return self.loss_weight[0] * loss_parse + self.loss_weight[1] * loss_edge + self.loss_weight[2] * loss_att_edge


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        intersection = predict * target
        score = (2. * intersection.sum(1) + self.smooth) / (predict.sum(1) + target.sum(1) + self.smooth)
        loss = 1 - score.sum() / predict.size(0)
        return loss