
import torch.nn.functional as F

from .metrics import FocalLoss, dice_loss, BCL, ThresholdBCL

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=2, alpha=0.25)

    bce = focal(predictions, target.long())
    dice = dice_loss(predictions, target.long())
    loss += bce + dice

    return loss

def hybrid_bcl(prediction, seg1, seg2, target, num_branches):
    """Calculating the loss"""
    loss = hybrid_loss(prediction, target)
    bcl = BCL()
    for i in range(num_branches-1):
        p1 = seg1[i].transpose(1,3)
        p2 = seg2[i].transpose(1,3)
        dist = F.pairwise_distance(p1,p2, keepdim=True).transpose(1,3)
        if dist.shape[2:] != target.shape[2:]:
            dist = F.interpolate(dist, size=target.shape[2:], mode='bilinear',align_corners=True)
        loss = loss + 0.1 * bcl(dist, target)

    return loss


CRITERION = {
    'hybrid':hybrid_loss,
    'hybrid_bcl':hybrid_bcl,
}

def DCAN_loss(arch='hybrid_bcl'):
    if arch in CRITERION.keys():
        return CRITERION[arch]
    else:
        raise NotImplementedError