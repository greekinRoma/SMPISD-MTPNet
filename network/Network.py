from network.backbones.mybackbone import MYPAFPN
from network.heads.my_heads import MYHead
from torch import nn
from .tools import *
from setting.read_setting import config
from network.layers.Pre_layer.TSSE import TSSE

class Network(nn.Module):
    def __init__(self,name='yolox_s',
                 strides=[8,16,32]):
        super().__init__()
        self.strides=config.strides
        self.backbone=None
        self.myhead=None
        self.decode=None
        self.grids=None
        self.training=False
        self.pre_layer = TSSE()
        self.choose_net(name)
        self.training = True
    def choose_net(self,name):
        if name == 'yolox_s':
            self.backbone = MYPAFPN(depth=0.33, width=0.5, in_channels=[256, 512, 1024], act='silu')
            self.myhead = MYHead(width=0.5, in_channels=config.in_channels, act='silu',num_classes=1)
        self.decode=anchor_free
        self.init_net()
        self.myhead.initialize_biases(1e-2)
    def init_net(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        for m in self.myhead.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
    def forward(self,input,training,use_pre_layer=False):
        if use_pre_layer:
            input = self.pre_layer(input)
        fpn_outs=self.backbone(input)
        dtype=input.type()
        if self.grids is None:
            self.grids = init_grids(fpn_outs, self.strides, dtype)
        outputs=self.myhead(fpn_outs)
        regs=[]
        for i, (stride, grid, output) in enumerate(zip(self.strides, self.grids, outputs)):
            regs.append(output[...,:4].clone())
            outputs[i] = self.decode(stride, grid, output)
        if training ==False:
            outputs = torch.cat(outputs, 1)
            return outputs
        else:
            strides=[]
            for grid,stride in zip(self.grids,self.strides):
                strides.append(torch.zeros(1,grid.shape[1]).fill_(stride).type(dtype))
            grids = torch.cat(self.grids, 1)
            outputs = torch.cat(outputs, 1)
            strides = torch.cat(strides,1)
            regs=torch.cat(regs,1)
            return outputs,grids,strides,regs
    def test(self,input):
        self.training =False
        input = self.pre_layer(input)
        fpn_outs=self.backbone(input)
        dtype=input.type()
        if self.grids is None:
            self.grids = init_grids(fpn_outs, self.strides, dtype)
        outputs=self.myhead(fpn_outs)
        regs=[]
        for i, (stride, grid, output) in enumerate(zip(self.strides, self.grids, outputs)):
            regs.append(output[...,:4].clone())
            outputs[i] = self.decode(stride, grid, output)
        outputs = torch.cat(outputs, 1)
        return outputs
if __name__=="__main__":
    net=Network('yolox_s')