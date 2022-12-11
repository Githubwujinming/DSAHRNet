import torch 
import torch.nn as nn
import torch.nn.functional as F
from .common import convout, get_encoder, BasicBlock
from .Decoder import *
from .CDCube import CDCubes


class CDNet(nn.Module):
    def __init__(self, num_blocks=[2,2,2,2], nc=2, output_stride=32,bilinear=True, num_branches=4, fuse=False) -> None:
        super().__init__()
        assert num_branches<=4 and num_branches>=1, 'out_depth must less than 4, greater than 1'
        # [32, 64, 160, 256]
        if output_stride >= 32:
            pool_list = [True, True, True, True]
        elif output_stride == 16:
            pool_list = [False, True, True, True]
        else:
            pool_list = [False, False, True, True]
        self.num_branches = num_branches
        self.encoder, self.channel_list = get_encoder('hrnet18')
        self.channel_list = self.channel_list[-num_branches:]
        # self.cbuilder = BASEBuildC(pool_list,self.channel_list[::-1],num_branches,bilinear=bilinear)
        self.cbuilder = BuildC_SCAT(pool_list,self.channel_list[::-1],num_branches,bilinear=bilinear)
        self.cube = CDCubes(self.channel_list,num_blocks[-num_branches:],pool_list,1,num_branches,bilinear,block=BasicBlock,cat=False, fuse=fuse)
        self.sca = SCAM(sum(self.channel_list[:num_branches]),sum(self.channel_list[:num_branches]))
        # self.convout1 = convout(sum(self.channel_list[:num_branches]),nc=nc)
        self.convoutc = convout(sum(self.channel_list[:num_branches]),nc=nc)
    def _init_weights(self,m):
       if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight)
          nn.init.constant_(m.bias, 0) 
       elif isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       elif isinstance(m, (nn.BatchNorm2d,nn.LayerNorm)):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)


    def _cat(self, c):
        cd = c[0]
        for i in range(1, self.num_branches):
            u = F.interpolate(c[i], size=cd.shape[2:], mode='bilinear',align_corners=True)
            cd = torch.cat((cd,u),dim=1)
        return cd


    def forward(self, pre_data, post_data):
        y1 = self.encoder(pre_data)[-self.num_branches:]
        y2 = self.encoder(post_data)[-self.num_branches:]
        y1, y2, c = self.cbuilder(y1,y2)
        y1, y2, c = self.cube(y1,y2,c)
        # y1 = self.convout1(y1)
        # y2 = self.convout1(y2)
        c = self._cat(c)
        c = self.sca(c)
        c = self.convoutc(c)
        return y1, y2, c
