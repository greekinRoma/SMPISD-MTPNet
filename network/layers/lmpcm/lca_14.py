import torch
from torch import nn
import numpy as np
from setting.read_setting import config as cfg
from network.network_blocks import BaseConv
import math
def put_text_pillow(image, text, position, font_path, font_size, color):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Draw text
    draw.text(position, text, font=font, fill=color)

    # Convert PIL image back to OpenCV image
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,out):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.out=out
        delta1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels//8,1)
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
        self.kernel1 = self.kernel1.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.out_conv_1 = nn.Conv2d(in_channels=in_channels + self.in_channels * 9, out_channels=out, kernel_size=1,stride=1)
        self.input_layer1 = nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels, kernel_size=1, stride=1)
        self.input_layer2 = nn.Conv2d(in_channels=in_channels, out_channels=self.in_channels, kernel_size=1, stride=1)
        self.layer1=nn.Conv2d(in_channels=self.in_channels*4,out_channels=self.in_channels,kernel_size=1,stride=1,groups=self.in_channels)
    def initialize_biases(self, prior_prob):
        b = self.out_conv_1.bias.view(1, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.out_conv_1.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def delta_conv(self,cen):
        cen=self.input_layer1(cen)
        delta1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", input=cen,groups=self.in_channels)
        delta8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", input=cen,groups=self.in_channels)
        dout = torch.concat([delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8], dim=1)
        return dout
    def square_conv(self,cen):
        b,_,w,h=cen.shape
        cen=self.input_layer2(cen)
        delta1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        delta8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", input=cen,
                                            groups=self.in_channels)
        square = torch.stack([delta1 * delta5, delta2 * delta6, delta3 * delta7, delta4 * delta8], dim=2)
        square = torch.sort(square, dim=2).values
        square = square.view(b, -1, w, h)
        sout = self.layer1(square)
        return sout
    def spatial_attention(self,inps):
        dout=self.delta_conv(inps)
        sout=self.square_conv(inps)
        outs=torch.concat([inps,sout,dout],1)
        outs=self.out_conv_1(outs)
        return outs
    def forward(self,cen,mas=None):
        outs=self.spatial_attention(cen)
        return outs