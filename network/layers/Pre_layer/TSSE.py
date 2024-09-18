import cv2
import torch
from torch import nn 
class TSSE(nn.Module):
    def __init__(self,shifts=[3,5,7]):
        super().__init__()
        self.shifts = shifts
    def circ_shift(self,cen, shift,hei,wid):
            B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
            B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
            B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
            B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
            B1_N = torch.concat([B1_NW, B1_NE], dim=3)
            B1_S = torch.concat([B1_SW, B1_SE], dim=3)
            B1 = torch.concat([B1_N, B1_S], dim=2)

            ######## B2 #########
            # old: A  =>  new: B
            #      B  =>       A
            B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
            B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
            B2 = torch.concat([B2_N, B2_S], dim=2)

            ######## B3 #########
            # old: AD  =>  new: CB
            #      BC  =>       DA
            B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
            B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
            B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
            B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
            B3_N = torch.concat([B3_NW, B3_NE], dim=3)
            B3_S = torch.concat([B3_SW, B3_SE], dim=3)
            B3 = torch.concat([B3_N, B3_S], dim=2)

            ######## B4 #########
            # old: AB  =>  new: BA
            B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
            B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
            B4 = torch.concat([B4_W, B4_E], dim=3)

            ######## B5 #########
            # old: AD  =>  new: CB
            #      BC  =>       DA
            B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
            B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
            B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
            B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
            B5_N = torch.concat([B5_NW, B5_NE], dim=3)
            B5_S = torch.concat([B5_SW, B5_SE], dim=3)
            B5 = torch.concat([B5_N, B5_S], dim=2)

            ######## B6 #########
            # old: A  =>  new: B
            #      B  =>       A
            B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
            B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
            B6 = torch.concat([B6_N, B6_S], dim=2)

            ######## B7 #########
            # old: AD  =>  new: CB
            #      BC  =>       DA
            B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
            B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
            B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
            B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
            B7_N = torch.concat([B7_NW, B7_NE], dim=3)
            B7_S = torch.concat([B7_SW, B7_SE], dim=3)
            B7 = torch.concat([B7_N, B7_S], dim=2)

            ######## B8 #########
            # old: AB  =>  new: BA
            B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
            B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
            B8 = torch.concat([B8_W, B8_E], dim=3)

            return B1, B2, B3, B4, B5, B6, B7, B8
    def ave_circ_shift(self,cen, shift,hei,wid):
        #对特征图进行平移，为了方便计算不同位置的对比度
        ######## B1 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        cen = torch.nn.functional.avg_pool2d(input=cen,kernel_size=(shift,shift),stride=1,padding=shift//2)
        B1_NW = cen[:,:, shift:, shift:]          # B1_NW is cen's SE
        B1_NE = torch.zeros_like(cen[:, :, shift:, :shift])      # B1_NE is cen's SW
        B1_SW = torch.zeros_like(cen[:, :, :shift, shift:])     # B1_SW is cen's NE
        B1_SE = torch.zeros_like(cen[:, :, :shift, :shift])          # B1_SE is cen's NW
        B1_N = torch.concat([B1_NW, B1_NE], dim=3)
        B1_S = torch.concat([B1_SW, B1_SE], dim=3)
        B1 = torch.concat([B1_N, B1_S], dim=2)
        ######## B2 #########
        # old: A  =>  new: B
        #      B  =>       A
        B2_N = cen[:,:,shift:, :]          # B2_N is cen's S
        B2_S = torch.zeros_like(cen[:, :, :shift, :])      # B2_S is cen's N
        B2 = torch.concat([B2_N, B2_S], dim=2)
        ######## B3 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B3_NW = torch.zeros_like(cen[:, :, shift:, wid-shift:])          # B3_NW is cen's SE
        B3_NE = cen[:,:,shift:, :wid-shift]      # B3_NE is cen's SW
        B3_SW = torch.zeros_like(cen[:, :, :shift, wid-shift:])      # B3_SW is cen's NE
        B3_SE = torch.zeros_like(cen[:, :, :shift, :wid-shift])          # B1_SE is cen's NW
        B3_N = torch.concat([B3_NW, B3_NE], dim=3)
        B3_S = torch.concat([B3_SW, B3_SE], dim=3)
        B3 = torch.concat([B3_N, B3_S], dim=2)
        ######## B4 #########
        # old: AB  =>  new: BA
        B4_W = torch.zeros_like(cen[:, :, :, wid-shift:])          # B2_W is cen's E
        B4_E = cen[:,:,:, :wid-shift]          # B2_E is cen's S
        B4 = torch.concat([B4_W, B4_E], dim=3)
        ######## B5 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B5_NW = torch.zeros_like(cen[:, :, hei-shift:, wid-shift:])          # B5_NW is cen's SE
        B5_NE = torch.zeros_like(cen[:, :, hei-shift:, :wid-shift])      # B5_NE is cen's SW
        B5_SW = torch.zeros_like(cen[:, :, :hei-shift, wid-shift:])      # B5_SW is cen's NE
        B5_SE = cen[:,:,:hei-shift, :wid-shift]          # B5_SE is cen's NW
        B5_N = torch.concat([B5_NW, B5_NE], dim=3)
        B5_S = torch.concat([B5_SW, B5_SE], dim=3)
        B5 = torch.concat([B5_N, B5_S], dim=2)
        ######## B6 #########
        # old: A  =>  new: B
        #      B  =>       A
        B6_N = torch.zeros_like(cen[:, :, hei-shift:, :])          # B6_N is cen's S
        B6_S = cen[:,:,:hei-shift, :]      # B6_S is cen's N
        B6 = torch.concat([B6_N, B6_S], dim=2)
        ######## B7 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B7_NW = torch.zeros_like(cen[:, :, hei-shift:, shift:])         # B7_NW is cen's SE
        B7_NE = torch.zeros_like(cen[:, :, hei-shift:, :shift])      # B7_NE is cen's SW
        B7_SW = cen[:,:,:hei-shift, shift:]      # B7_SW is cen's NE
        B7_SE = torch.zeros_like(cen[:, :, :hei-shift, :shift])        # B7_SE is cen's NW
        B7_N = torch.concat([B7_NW, B7_NE], dim=3)
        B7_S = torch.concat([B7_SW, B7_SE], dim=3)
        B7 = torch.concat([B7_N, B7_S], dim=2)
        ######## B8 #########
        # old: AB  =>  new: BA
        B8_W = cen[:,:,:, shift:]          # B8_W is cen's E
        B8_E = torch.zeros_like(cen[:, :, :, :shift])          # B8_E is cen's S
        B8 = torch.concat([B8_W, B8_E], dim=3)
        return B1, B2, B3, B4, B5, B6, B7, B8,cen
    def cal_pcm(self,cen, shift,wid,hei):
        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift,hei=hei,wid=wid)
        delta1=B1-cen
        delta2=B2-cen
        delta3=B3-cen
        delta4=B4-cen
        delta5=B5-cen
        delta6=B6-cen
        delta7=B7-cen
        delta8=B8-cen
        s = torch.stack([delta1*delta5,delta2*delta6,delta3*delta7,delta4*delta8],0)
        s = torch.sort(s,0).values
        outs = torch.mean(s[:2], 0)
        ####################################
        tmps=(torch.abs(delta1)+torch.abs(delta2)+torch.abs(delta3)+torch.abs(delta4)+torch.abs(delta5)+torch.abs(delta6)+torch.abs(delta7)+torch.abs(delta8))/8.
        T1, T2, T3, T4, T5, T6, T7, T8,_ = self.ave_circ_shift(tmps, shift=shift*3,hei=hei,wid=wid)
        Ts = torch.stack([T1,T2,T3,T4,T5,T6,T7,T8],0)
        out_mask=torch.min(Ts,0).values
        out_mask= 1/(out_mask+1)
        outs=outs*out_mask
        return outs
    def forward(self,src):
        src = torch.mean(src,dim=1,keepdim=True)
        tmps=[]
        _,_,wid,hei =src.shape
        for shift in self.shifts:
            tmps.append(self.cal_pcm(cen=src,shift=shift,wid=wid,hei=hei))
        tmps=torch.stack(tmps,0)
        dst=torch.max(tmps,0).values
        dst=torch.concat([dst%256,torch.clip(dst,min=0)//256,src],dim=1)
        return dst