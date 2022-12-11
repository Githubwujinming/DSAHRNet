import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

#Focal Loss
def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha

# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1).long()
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)# 求每个像素对应的预测值，比如一个标记为变化的像素点则取变化的预测值。
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

#miou loss
from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)
def hybrid_loss(pred, target):
    """Calculating the loss"""
    if pred.shape[2] != target.shape[2]:
        target = F.interpolate(target, size=pred.shape[2:], mode='bilinear',align_corners=True)
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=2, alpha=0.25)

    bce = focal(pred, target.long())
    dice = dice_loss(pred, target.long())
    loss += bce + dice

    return loss

def change_former_loss(preds, target, weights=[0.5, 0.5, 0.5, 0.8, 1.0]):
    loss = 0
    for p,w in zip(preds,weights):
        loss = loss + w*hybrid_loss(p, target)
    return loss
