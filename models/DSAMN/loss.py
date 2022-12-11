from .utils import make_one_hot
from .BCL import BCL
from .DiceLoss import DiceLoss
import torch
def dsamn_loss(prob, ds2, ds3, label, device, wDice=0.1):
    label = make_one_hot(label.long(),2).to(device)
    # label = make_one_hot(label.unsqueeze(0).long(),2).squeeze(0)
    CDcriterion = BCL().to(device, dtype=torch.float)
    CDcriterion1 = DiceLoss().to(device, dtype=torch.float)
    #Diceloss
    dsloss2 = CDcriterion1(ds2, label)
    dsloss3 = CDcriterion1(ds3, label)

    Dice_loss = 0.5*(dsloss2+dsloss3)

    # contrative loss
    label = torch.argmax(label, 1).unsqueeze(1).float()
    CT_loss = CDcriterion(prob, label)

    # CD loss
    CD_loss =  CT_loss + wDice *Dice_loss
    return CD_loss