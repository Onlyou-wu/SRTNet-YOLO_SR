import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sr_decoder_noBN_noD import Decoder
from models.SCNet_arch import SCNet11


class DeepLab(nn.Module):
    def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d
        self.sr_decoder = Decoder(c1,c2)

        self.SCNet = SCNet11(num_in_ch=ch, num_out_ch=ch, num_feat=64, num_block=16, upscale=2)

        self.factor = factor


    def forward(self, low_level_feat,x): 
        x_sr= self.sr_decoder(x, low_level_feat, self.factor)   
        x_sr_up = self.SCNet(x_sr)


        return x_sr_up   


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


