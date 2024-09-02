import torch
from torch import nn
import numpy as np
from setting.read_setting import config as cfg
from network.network_blocks import BaseConv
import math
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,out):
        super().__init__()
        #The hyper parameters settting
        self.convs_list=nn.ModuleList()
        self.out=out
        delta1=np.array([[[-1, 0, 0], [0, 0, 0], [0, 0, 0]], 
                         [[0, -1, 0], [0, 0, 0], [0, 0, 0]], 
                         [[0, 0, -1], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
        self.shifts=[1,5]
        self.scale=torch.nn.Parameter(torch.zeros(len(self.shifts)))
        self.hidden_channels=max(self.in_channels//8,1)//len(self.shifts)
        #The Process the of extraction of outcome
        if cfg.use_cuda:
            self.kernel1 = torch.Tensor(w1).cuda()
            self.kernel2 = torch.Tensor(w2).cuda()
            self.kernel3 = torch.Tensor(w3).cuda()
            self.kernel4 = torch.Tensor(w4).cuda()
            self.kernel5 = torch.Tensor(w5).cuda()
            self.kernel6 = torch.Tensor(w6).cuda()
            self.kernel7 = torch.Tensor(w7).cuda()
            self.kernel8 = torch.Tensor(w8).cuda()
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
            self.kernel3 = torch.Tensor(w3)
            self.kernel4 = torch.Tensor(w4)
            self.kernel5 = torch.Tensor(w5)
            self.kernel6 = torch.Tensor(w6)
            self.kernel7 = torch.Tensor(w7)
            self.kernel8 = torch.Tensor(w8)
        self.kernel1 = self.kernel1.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.hidden_channels, 1, 1, 1).contiguous()
        #After Extraction, we analyze the outcome of the extraction. 
        self.num_layer=9
        self.out_conv=nn.Conv2d(in_channels=self.hidden_channels*len(self.shifts)*self.num_layer,out_channels=out,kernel_size=1,stride=1)
        self.act=nn.Softmax(dim=2)
        self.input_layers=nn.ModuleList()
        self.layers_1=nn.ModuleList()
        self.layers_2=nn.ModuleList()
        self.layers_3=nn.ModuleList()
        for shift in self.shifts:
            kernel=max(1,shift-2)
            self.input_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=self.hidden_channels,kernel_size=kernel,stride=1,padding='same'))
            self.layers_1.append(nn.Conv2d(in_channels=self.hidden_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,groups=self.num_layer,bias=False))
            self.layers_2.append(nn.Conv2d(in_channels=self.hidden_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,groups=self.num_layer,bias=False))
            self.layers_3.append(nn.Conv2d(in_channels=self.hidden_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,groups=self.num_layer,bias=False))
    def initialize_biases(self, prior_prob):
        b = self.out_conv.bias.view(1, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.out_conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def Extract_layer(self,cen,i):
        cen=self.input_layers[i](cen)
        delta1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        delta8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
        deltas = torch.concat([delta1,delta2,delta3,delta4,delta5,delta6,delta7,delta8,cen],1)
        return deltas
    def Encode_layer(self,delats,i):
        return delats
    def Analyze_layer(self,deltas,i):
        b,c,w,h=deltas.shape
        out1=self.layers_1[i](deltas).view(b,1,self.num_layer,c//self.num_layer,w,h).contiguous()
        out2=self.layers_2[i](deltas).view(b,self.num_layer,1,c//self.num_layer,w,h).contiguous()
        out3=self.layers_3[i](deltas).view(b,1,self.num_layer,c//self.num_layer,w,h).contiguous()
        attention=self.act(torch.sum(out1*out2*self.scale[i],dim=3,keepdim=True))
        # print(attention[0,0,:,0,0,0])
        out=torch.sum(attention*out3,2).view(b,-1,w,h).contiguous()
        return out
    def forward(self,cen):
        outs=[]
        for i in range(len(self.shifts)):
            deltas=self.Extract_layer(cen=cen,i=i)
            out=self.Analyze_layer(deltas=deltas,i=i)
            outs.append(out)
        outs=torch.concat(outs,1)
        return self.out_conv(outs)