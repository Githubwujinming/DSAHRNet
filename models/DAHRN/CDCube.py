import torch 
import torch.nn as nn
import torch.nn.functional as F
from .common import  BasicBlock
from .Decoder import *
bn_mom = 0.0003
class HRModule(nn.Module):
    '''
    num_branches : how many branches we have
    blocks: which type we want to use
    num_blocks: how many blocks for each branch
    num_inchannels: inchannel for each block
    num_channels: here we set num_inchannels
    multi_scale_output: wether span multi scale output
    '''
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, multi_scale_output=True, norm_layer=None, fuse=True):
        super().__init__()

        #[32, 64, 160, 256]
        self._check_branches(
            num_branches, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.fuse = fuse
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        if fuse:
            self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.GELU()

    def _check_branches(self, num_branches, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample=downsample, BatchNorm=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], BatchNorm=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        if self.fuse:
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        y = y + F.interpolate(
                            self.fuse_layers[i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear',
                            align_corners=True
                            )
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
            return x_fuse
        else:
            return x
class CDCube(nn.Module):
    def __init__(self, num_branches, blocks,  num_inchannels,pool_list, num_blocks=[2,2,3,3],
    multi_scale_output=True, norm_layer=None, bilinear=True, fuse=True) -> None:
        super().__init__()
        #[32, 64, 160, 256]
        assert num_branches > 0 and num_branches <=4, 'num_branches must in (0,4]'
        self.num_branches = num_branches
        # self.channel_list = num_inchannels[:num_branches]
        self.stageY = HRModule(num_branches, blocks, num_blocks, num_inchannels, # num_block [ 1,1,2,2]
                 num_inchannels, multi_scale_output, norm_layer, fuse=fuse)
        self.stageC = HRModule(num_branches, blocks, num_blocks, num_inchannels,
                 num_inchannels, multi_scale_output, norm_layer, fuse=fuse)
        self.yc = YC_SCAT(pool_list,num_inchannels[::-1],num_branches=num_branches,bilinear=bilinear)

    def _branch(self,x1, x2, c):
        x1 = self.stageY(x1)
        x2 = self.stageY(x2)
        c = self.stageC(c)
        return x1, x2, c

    def forward(self,x1, x2, c):
        x1 = x1[-self.num_branches:]
        x2 = x2[-self.num_branches:]
        c = c[-self.num_branches:]
        x1, x2, c = self._branch(x1, x2, c)
        x1, x2, c = self.yc(x1, x2, c) 
        return x1, x2, c


class CDCubes(nn.Module):
    def __init__(self,channel_list, num_blocks=[2,2,3,3], pool_list=[True, True, True, True],cube_num=3, num_branches=4, bilinear=False, norm=nn.BatchNorm2d, block=BasicBlock, cat=True, fuse=True) -> None:
        super().__init__()
        #[32, 64, 160, 256]
        self.cube_num = cube_num
        self.cubes = nn.ModuleList([CDCube(num_branches, block,  channel_list, pool_list, num_blocks,
                             bilinear=bilinear, norm_layer=norm, fuse=fuse) for i in range(cube_num)])
        self.num_branches = num_branches
        self.cat = cat
    def _cat(self,x1, x2, c):
        y1, y2, cd = x1[0], x2[0], c[0]
        for i in range(1, self.num_branches):
            if y1.shape[2:] != x1[i].shape[2:]:
                u1 = F.interpolate(x1[i], size=y1.shape[2:], mode='bilinear',align_corners=True)
                u2 = F.interpolate(x2[i], size=y2.shape[2:], mode='bilinear',align_corners=True)
                u = F.interpolate(c[i], size=cd.shape[2:], mode='bilinear',align_corners=True)
            y1 = torch.cat((y1,u1),dim=1)
            y2 = torch.cat((y2,u2),dim=1)
            cd = torch.cat((cd,u),dim=1)
        return y1, y2, cd

    def forward(self, y1, y2, c):
        for i in range(self.cube_num):
            y1, y2, c = self.cubes[i](y1,y2,c)
        if self.cat:
            y1, y2, c = self._cat(y1, y2, c) 
        return y1, y2, c



