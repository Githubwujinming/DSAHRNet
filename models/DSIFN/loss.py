
import torch
import torch.nn as nn
import torch.nn.functional as F

def cd_loss(input,target):
    if input.shape[2] != target.shape[2]:
        target = F.interpolate(target, size=input.shape[2:], mode='bilinear',align_corners=True)
    bce_loss = nn.BCELoss()
    bce_loss = bce_loss(torch.sigmoid(input),torch.sigmoid(target))

    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dic_loss = 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

    return  dic_loss + bce_loss

def multi_supvised_loss(inputs, target, w=1):
    loss = 0
    for input in inputs:
        loss = loss + w*cd_loss(input, target)
    return loss