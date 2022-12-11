
from .metrics import FocalLoss, dice_loss



def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=2, alpha=0.25)

    bce = focal(predictions, target.long())
    dice = dice_loss(predictions, target.long())
    loss += bce + dice

    return loss

