import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import net.base_net as base_net


###############################################################################
# define spatially adaptive normalized generator
# input: boundary and apperance latent vector
###############################################################################


def defineSPADEGenerator(input_nc, output_nc, ngf, norm='instance', use_dropout=False, init_type='normal',
               init_gain=0.02, gpu_ids=[], latent_chl=1024, up_mode='NF'):
    norm_layer = base_net.get_norm_layer(norm_type=norm)
    net = SPADEGenerator(input_nc, output_nc, ngf,
                            norm_layer=norm_layer, latent_chl=latent_chl, up_mode=up_mode)
    return base_net.init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

# class BasicSPADE(nn.Module):
#     def __init__(self, norm_layer, input_nc, planes):
#         super(BasicSPADE, self).__init__()
#         self.norm = norm_layer(planes, affine=False)
#
#         self.conv_weight1=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_bias1=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_weight2=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_bias2=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_weight3=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_bias3=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_weight4=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#         self.conv_bias4=nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)
#
#         self.conv_weight=nn.Conv2d(input_nc, planes, kernel_size=3, stride=1, padding=1)
#         self.conv_bias=nn.Conv2d(input_nc, planes, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x, bound):
#         out = self.norm(x)
#
#         weight_norm1 = self.conv_weight1(bound)
#         bias_norm1 = self.conv_bias1(bound)
#         weight_norm2 = self.conv_weight2(weight_norm1)
#         bias_norm2 = self.conv_bias2(bias_norm1)
#         weight_norm3 = self.conv_weight3(weight_norm2)
#         bias_norm3 = self.conv_bias3(bias_norm2)
#         weight_norm4 = self.conv_weight4(weight_norm3)
#         bias_norm4 = self.conv_bias4(bias_norm3)
#
#         weight_norm = self.conv_weight(weight_norm4)
#         bias_norm = self.conv_bias(bias_norm4)
#
#         out = out * weight_norm + bias_norm
#         return out

class BasicSPADE(nn.Module):
    def __init__(self, norm_layer, input_nc, planes):
        super(BasicSPADE, self).__init__()
        self.conv_weight = nn.Conv2d(input_nc, planes, kernel_size=3, stride=1, padding=1)
        self.conv_bias = nn.Conv2d(input_nc, planes, kernel_size=3, stride=1, padding=1)
        self.norm = norm_layer(planes, affine=False)

    def forward(self, x, bound):
        out = self.norm(x)
        weight_norm = self.conv_weight(bound)
        bias_norm = self.conv_bias(bound)
        out = out * weight_norm + bias_norm
        return out


class ResBlkSPADE(nn.Module):
    def __init__(self, norm_layer, input_nc, planes, conv_kernel_size=1, padding=0):   # todo: change conv kernel size, kernel=3, padding=1 or kernel=1, padding=0
        super(ResBlkSPADE, self).__init__()
        self.spade1 = BasicSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spade2 = BasicSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spade_res = BasicSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=planes)
        self.conv_res = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self, x, bound):
        out = self.spade1(x, bound)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spade2(out, bound)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        residual = self.spade_res(residual, bound)
        residual = self.relu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return out

# Defines the generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class SPADEGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm2d, latent_chl=1024, up_mode='NF'):
        super(SPADEGenerator, self).__init__()

        layers = []
        self.up_mode = up_mode

        self.up1 = nn.ConvTranspose2d(in_channels=latent_chl, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)

        if self.up_mode == 'convT':
            self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            self.up5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
            self.up6 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
            self.up7 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
            self.up8 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
            # self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            # self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            # self.up5 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            # self.up6 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
            # self.up7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
            # self.up8 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        elif self.up_mode == 'NF':
            self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up8 = nn.Upsample(scale_factor=2, mode='nearest')

        self.spade_blc3 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        self.spade_blc4 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        self.spade_blc5 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=1024+512)
        self.spade_blc6 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512+256)
        self.spade_blc7 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=256+128)
        self.spade_blc8 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=128+64)

        self.conv5 = nn.Conv2d(in_channels=1024+512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=512+256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=256+128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # self.spade_blc3 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        # self.spade_blc4 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        # self.spade_blc5 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        # self.spade_blc6 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=256)
        # self.spade_blc7 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=128)
        # self.spade_blc8 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=64)
        #
        # self.conv5 = nn.Conv2d(in_channels=1024 + 512, out_channels=512, kernel_size=1, stride=1, padding=0)
        # self.conv6 = nn.Conv2d(in_channels=512 + 512, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.conv7 = nn.Conv2d(in_channels=256 + 256, out_channels=128, kernel_size=1, stride=1, padding=0)
        # self.conv8 = nn.Conv2d(in_channels=128 + 128, out_channels=64, kernel_size=1, stride=1, padding=0)


        self.same = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, input, latent_z, decoder_result): # input: bound, batch_size*17*256*256
        bound128 = F.interpolate(input, scale_factor=0.5)
        bound64 = F.interpolate(bound128, scale_factor=0.5)
        bound32 = F.interpolate(bound64, scale_factor=0.5)
        bound16 = F.interpolate(bound32, scale_factor=0.5)
        bound8 = F.interpolate(bound16, scale_factor=0.5)
        bound4 = F.interpolate(bound8, scale_factor=0.5)

        x_up1 = self.up1(latent_z)
        x_up2 = self.up2(x_up1)

        x_up3 = self.spade_blc3(x_up2, bound4) # 4*4 bound
        x_up3 = self.up3(x_up3)

        x_up4 = self.spade_blc4(x_up3, bound8) # 8*8 bound
        x_up4 = self.up4(x_up4)

        x_up5 = self.spade_blc5(torch.cat([x_up4, decoder_result[0]], 1), bound16) # 16*16 bound
        x_up5 = self.conv5(x_up5)
        x_up5 = self.up5(x_up5)

        x_up6 = self.spade_blc6(torch.cat([x_up5, decoder_result[1]], 1), bound32) # 32*32 bound
        x_up6 = self.conv6(x_up6)
        x_up6 = self.up6(x_up6)

        x_up7 = self.spade_blc7(torch.cat([x_up6, decoder_result[2]], 1), bound64) # 64*64 bound
        x_up7 = self.conv7(x_up7)
        x_up7 = self.up7(x_up7)

        x_up8 = self.spade_blc8(torch.cat([x_up7, decoder_result[3]], 1), bound128) # 128*128 bound
        x_up8 = self.conv8(x_up8)
        x_up8 = self.up8(x_up8)


        # x_up5 = self.conv5(torch.cat([x_up4, decoder_result[0]], 1))
        # x_up5 = self.spade_blc5(x_up5, bound16)  # 16*16 bound
        # x_up5 = self.up5(x_up5)
        #
        # x_up6 = self.conv6(torch.cat([x_up5, decoder_result[1]], 1))
        # x_up6 = self.spade_blc6(x_up6, bound32)  # 16*16 bound
        # x_up6 = self.up6(x_up6)
        #
        # x_up7 = self.conv7(torch.cat([x_up6, decoder_result[2]], 1))
        # x_up7 = self.spade_blc7(x_up7, bound64)  # 16*16 bound
        # x_up7 = self.up7(x_up7)
        #
        # x_up8 = self.conv8(torch.cat([x_up7, decoder_result[3]], 1))
        # x_up8 = self.spade_blc8(x_up8, bound128)  # 16*16 bound
        # x_up8 = self.up8(x_up8)


        x_out = self.same(x_up8)
        x_out = self.tanh(x_out)

        return x_out



# # define upSample Module
# class UpSampleBlock(nn.Module):
#     def __init__(self, input_nc, output_nc,
#                  outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         super(UpSampleBlock, self).__init__()
#
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(output_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose2d(input_nc, output_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             up = [uprelu, upconv, nn.Tanh()]
#
#         elif innermost:
#             upconv = nn.ConvTranspose2d(input_nc, output_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             up = [uprelu, upconv, upnorm]
#
#         else:
#             upconv = nn.ConvTranspose2d(input_nc, output_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             up = [uprelu, upconv, upnorm]
#             if use_dropout:
#                 up = up + [nn.Dropout(0.5)]
#
#         self.up = nn.Sequential(*up)
#
#     def forward(self, x):
#         return self.up(x)


