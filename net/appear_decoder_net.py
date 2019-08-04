import torch as th
from torch import nn
import net.base_net as base_net
###############################################################################
# define
###############################################################################


def defineAppDec(input_nc, size_=256, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = base_net.get_norm_layer(norm_type=norm)
    if 128 == size_:
        net = appearDec128(input_nc, norm_layer=norm_layer, size_=size_)
    elif 256 == size_:
        net = appearDec(input_nc, norm_layer=norm_layer, size_=size_)
    return base_net.init_net(net, init_type, init_gain, gpu_ids)


class appearDec(nn.Module):
    def __init__(self, input_c, norm_layer, size_=256):
        super(appearDec, self).__init__()
        # input 3x256x256
        # encoder
        layers = []

        channel_list = [1024, 1024, 1024, 1024]  
        c0 = 1024
        for cc in channel_list:
            layers.append(nn.ConvTranspose2d(c0, cc, 4, 2, 1))
            layers.append(norm_layer(cc))
            layers.append(nn.ReLU(True))
            c0 = cc
        self.decoder16 = nn.Sequential(*layers)

        self.decoder32 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), norm_layer(512), nn.ReLU(True))
        self.decoder64 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), norm_layer(256), nn.ReLU(True))
        self.decoder128 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), norm_layer(128), nn.ReLU(True))
        layers = []
        layers.append(nn.ConvTranspose2d(128, 3, 4, 2, 1))
        layers.append(nn.Tanh())
        self.decoder256 = nn.Sequential(*layers)


    def forward(self, input):
        out16 = self.decoder16(input)
        out32 = self.decoder32(out16)
        out64 = self.decoder64(out32)
        out128 = self.decoder128(out64)
        out256 = self.decoder256(out128)
        return out16, out32, out64, out128, out256

class appearDec128(nn.Module):
    def __init__(self, input_c, norm_layer, size_=256):
        super(appearDec128, self).__init__()
        # input 3x256x256
        # encoder
        layers = []

        channel_list = [1024, 1024, 1024]  
        c0 = 1024
        for cc in channel_list:
            layers.append(nn.ConvTranspose2d(c0, cc, 4, 2, 1))
            layers.append(norm_layer(cc))
            layers.append(nn.ReLU(True))
            c0 = cc
        self.decoder8 = nn.Sequential(*layers)

        self.decoder16 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), norm_layer(512), nn.ReLU(True))
        self.decoder32 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), norm_layer(256), nn.ReLU(True))
        self.decoder64 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), norm_layer(128), nn.ReLU(True))
        layers = []
        layers.append(nn.ConvTranspose2d(128, 3, 4, 2, 1))
        layers.append(nn.Tanh())
        self.decoder128 = nn.Sequential(*layers)


    def forward(self, input):
        out8 = self.decoder8(input)
        out16 = self.decoder16(out8)
        out32 = self.decoder32(out16)
        out64 = self.decoder64(out32)
        out128 = self.decoder128(out64)
        return out8, out16, out32, out64, out128