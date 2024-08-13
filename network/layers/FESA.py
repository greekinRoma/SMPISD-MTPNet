from torch import nn
class FESA(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.xmax_pooling1d=nn.AdaptiveMaxPool1d(output_size=1,)