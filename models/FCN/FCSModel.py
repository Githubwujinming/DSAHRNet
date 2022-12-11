from .FCEF import FCEF
from .FCSiamConc import FCSiamConc
from .FCSiamDiff import FCSiamDiff
def getFCSModel(net='FCEF',in_c=3):
    if net=='FCEF':
        print('using FCEF model')
        model = FCEF(in_c*2)
    elif net=='FCSC':
        print('using FCSiamConc model')
        model = FCSiamConc(in_c)
    elif net=='FCSD':
        print('using FCSiamDiff model')
        model = FCSiamDiff(in_c)
    else:
        NotImplementedError('no such  FCS model!!')
    return model
