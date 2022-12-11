import torch 
import torch.nn as nn

from .common import *

class BASEBuildC(nn.Module):
    def __init__(self, pool_list=[True, True, True, True], channel_list=[256,160, 64, 32], num_branches=4, mlp_ratio=4., drop=0.,drop_path=0.3, bilinear=False, act_layer=nn.GELU,bn_mom = 0.0003,) -> None:
        super().__init__()
        self.depth = num_branches
        self.cdblock0 = CAT(channel_list[-1],channel_list[-1],channel_list[-1], bilinear=bilinear,upsample=False)
        self.cddecoder0 = CAT(channel_list[-2],channel_list[-1],channel_list[-1], bilinear=bilinear,upsample=pool_list[-2])
        for i in range(1, num_branches):#  2, 3
            cdblock = CAT(channel_list[i-1],channel_list[i-1],channel_list[i-1], bilinear=bilinear,upsample=False)
            cddecoder = CAT(channel_list[max(i-2,0)], channel_list[i-1], channel_list[i-1],bilinear=bilinear,upsample=pool_list[i])
            decoder = CAT(channel_list[i-1],channel_list[i],channel_list[i],upsample=pool_list[i],bilinear=bilinear)
            setattr(self,f"cdblock{i}", cdblock)
            setattr(self,f"decoder{i}", decoder)
            setattr(self,f"cddecoder{i}", cddecoder)
    def forward(self, y1, y2):
        cs = []
        c_pre = None
        for i in range(1,self.depth):#  2, 3 
            cdblock = getattr(self,f'cdblock{i}')
            decoder = getattr(self,f'decoder{i}')
            cddecoder = getattr(self,f'cddecoder{i}')
            c = cdblock(y1[-i], y2[-i])
            if c_pre is not None:
                c = cddecoder(c_pre, c)
            cs.insert(0, c)
            y1[-i-1] = decoder(y1[-i], y1[-i-1])
            y2[-i-1] = decoder(y2[-i], y2[-i-1])
            c_pre = c
        c = self.cdblock0(y1[0], y2[0])
        c = self.cddecoder0(c_pre, c)
        cs.insert(0, c)
        return y1, y2, cs


class BuildC_SCAT(nn.Module):
    def __init__(self, pool_list=[True, True, True, True], channel_list=[256,160, 64, 32], num_branches=4, mlp_ratio=4., drop=0.,drop_path=0.3, bilinear=False, act_layer=nn.GELU,bn_mom = 0.0003,) -> None:
        super().__init__()
        self.depth = num_branches
        self.cdblock0 = SCAT(channel_list[-1],channel_list[-1],channel_list[-1], bilinear=bilinear,upsample=False)
        self.cddecoder0 = SCAT(channel_list[-2],channel_list[-1],channel_list[-1], bilinear=bilinear,upsample=pool_list[-2])
        for i in range(1, num_branches):#  2, 3
            cdblock = SCAT(channel_list[i-1],channel_list[i-1],channel_list[i-1], bilinear=bilinear,upsample=False)
            cddecoder = SCAT(channel_list[max(i-2,0)], channel_list[i-1], channel_list[i-1],bilinear=bilinear,upsample=pool_list[i])
            decoder = SCAT(channel_list[i-1],channel_list[i],channel_list[i],upsample=pool_list[i],bilinear=bilinear)
            setattr(self,f"cdblock{i}", cdblock)
            setattr(self,f"decoder{i}", decoder)
            setattr(self,f"cddecoder{i}", cddecoder)
    def forward(self, y1, y2):
        cs = []
        c_pre = None
        for i in range(1,self.depth):#  2, 3 
            cdblock = getattr(self,f'cdblock{i}')
            decoder = getattr(self,f'decoder{i}')
            cddecoder = getattr(self,f'cddecoder{i}')
            c = cdblock(y1[-i], y2[-i])
            if c_pre is not None:
                c = cddecoder(c_pre, c)
            cs.insert(0, c)
            y1[-i-1] = decoder(y1[-i], y1[-i-1])
            y2[-i-1] = decoder(y2[-i], y2[-i-1])
            c_pre = c
        c = self.cdblock0(y1[0], y2[0])
        c = self.cddecoder0(c_pre, c)
        cs.insert(0, c)
        return y1, y2, cs


class YC_CAT(nn.Module):
    def __init__(self, pool_list=[True, True, True, True], channel_list=[32,64,160,256], num_branches=4, mlp_ratio=4., drop=0.,drop_path=0.3, bilinear=False, act_layer=nn.GELU,bn_mom = 0.0003) -> None:
        super().__init__()
        #[256, 160, 64, 32]
        self.depth = num_branches
        self.cdblock0 = TCAT(channel_list[-1])
        self.cddecoder0 = CAT(channel_list[-2],channel_list[-1],channel_list[-1], bilinear=bilinear,upsample=pool_list[-2])
        for i in range(1, num_branches):# 1, 2, 3
            cdblock = TCAT(channel_list[i-1])
            cddecoder = CAT(channel_list[i-1],channel_list[i],channel_list[i],upsample=pool_list[i],bilinear=bilinear)
            decoder = CAT(channel_list[i-1],channel_list[i],channel_list[i],upsample=pool_list[i],bilinear=bilinear)
            setattr(self,f"cdblock{i}", cdblock)
            setattr(self,f"decoder{i}", decoder)
            setattr(self,f"cddecoder{i}", cddecoder)
    def forward(self, y1, y2, c):
        for i in range(1,self.depth):# 1, 2, 3 
            cdblock = getattr(self,f'cdblock{i}')
            decoder = getattr(self,f'decoder{i}')
            cddecoder = getattr(self,f'cddecoder{i}')
            c[-i] = cdblock(y1[-i], y2[-i], c[-i])
            y1[-i-1] = decoder(y1[-i], y1[-i-1])
            y2[-i-1] = decoder(y2[-i], y2[-i-1])
            c[-i-1] = cddecoder(c[-i], c[-i-1])
        c[0] = self.cdblock0(y1[0], y2[0],c[0])
        c[0] = self.cddecoder0(c[1], c[0])
        return y1, y2, c


class YC_SCAT(nn.Module):
    def __init__(self, pool_list=[True, True, True, True], channel_list=[32,64,160,256], num_branches=4, mlp_ratio=4., drop=0.,drop_path=0.3, bilinear=False, act_layer=nn.GELU,bn_mom = 0.0003) -> None:
        super().__init__()
        #[256, 160, 64, 32]
        self.depth = num_branches
        self.cdblock0 = TSCAT(channel_list[-1])
        self.cddecoder0 = SCAT(channel_list[-2],channel_list[-1],channel_list[-1], bilinear=bilinear,upsample=pool_list[-2])
        for i in range(1, num_branches):# 1, 2, 3
            cdblock = TSCAT(channel_list[i-1])
            cddecoder = SCAT(channel_list[i-1],channel_list[i],channel_list[i],upsample=pool_list[i],bilinear=bilinear)
            decoder = SCAT(channel_list[i-1],channel_list[i],channel_list[i],upsample=pool_list[i],bilinear=bilinear)
            setattr(self,f"cdblock{i}", cdblock)
            setattr(self,f"decoder{i}", decoder)
            setattr(self,f"cddecoder{i}", cddecoder)
    def forward(self, y1, y2, c):
        for i in range(1,self.depth):# 1, 2, 3 
            cdblock = getattr(self,f'cdblock{i}')
            decoder = getattr(self,f'decoder{i}')
            cddecoder = getattr(self,f'cddecoder{i}')
            c[-i] = cdblock(y1[-i], y2[-i], c[-i])
            y1[-i-1] = decoder(y1[-i], y1[-i-1])
            y2[-i-1] = decoder(y2[-i], y2[-i-1])
            c[-i-1] = cddecoder(c[-i], c[-i-1])
        c[0] = self.cdblock0(y1[0], y2[0],c[0])
        c[0] = self.cddecoder0(c[1], c[0])
        return y1, y2, c

