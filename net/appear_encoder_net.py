import torch as th
from torch import nn
import net.base_net as base_net
###############################################################################
# define
###############################################################################


def defineAppEnc(input_nc, size_=256, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], conv_k=4):
    net = None
    norm_layer = base_net.get_norm_layer(norm_type=norm)

    net = appearEnc(input_nc, norm_layer=norm_layer, size_=size_, conv_k=conv_k)
    return base_net.init_net(net, init_type, init_gain, gpu_ids)


# class appearEnc(nn.Module):
#     def __init__(self, input_c, norm_layer, size_=256, conv_k=4):
#         super(appearEnc, self).__init__()
#         # input 3x256x256
#         # encoder
#         channel_list = [128, 256, 512, 1024, 1024, 1024]
#         c0 = 64
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(input_c, c0, conv_k, 2, 1),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(c0, channel_list[0], conv_k, 2, 1),
#             norm_layer(channel_list[0]),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(channel_list[0], channel_list[1], conv_k, 2, 1),
#             norm_layer(channel_list[1]),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(channel_list[1], channel_list[2], conv_k, 2, 1),
#             norm_layer(channel_list[2]),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(channel_list[2], channel_list[3], conv_k, 2, 1),
#             norm_layer(channel_list[3]),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(channel_list[3], channel_list[4], conv_k, 2, 1),
#             norm_layer(channel_list[4]),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(channel_list[4], channel_list[5], conv_k, 2, 1),
#             norm_layer(channel_list[5]),
#             nn.LeakyReLU(0.2)
#         )
#         self.mean = nn.Conv2d(1024, 1024, conv_k, 2, 1)

class appearEnc(nn.Module):
    def __init__(self, input_c, norm_layer, size_=256, conv_k=4):
        super(appearEnc, self).__init__()
        # input 3x256x256
        # encoder
        layers = []
        channel_list = [128, 256, 512, 1024, 1024, 1024]

        c0 = 64
        layers.append(nn.Conv2d(input_c, c0, conv_k, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        for cc in channel_list:
            layers.append(nn.Conv2d(c0, cc, conv_k, 2, 1))
            layers.append(norm_layer(cc))
            layers.append(nn.LeakyReLU(0.2))
            c0 = cc
        self.encoder = nn.Sequential(*layers)
        # mean
        layers = []
        layers.append(nn.Conv2d(1024, 1024, conv_k, 2, 1))
        # layers.append(nn.ReLU())
        self.mean = nn.Sequential(*layers)
        # self.logvar = nn.Sequential(*layers)



    def sample_z(self, z_mu):
        z_std = 1.0
        eps = th.randn(z_mu.size()).type_as(z_mu) # random number in [0,1]
        return z_mu + z_std * eps

    # def sample_z(self, z_mu, z_logvar):
    #     z_std = th.exp(0.5 * z_logvar)
    #     eps = th.randn_like(z_std)
    #     return z_mu + z_std * eps

    def kl_loss(self, z_mu):
        #kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        z_var = th.ones(z_mu.size()).type_as(z_mu) # [batch_size, 1024, 1, 1]
        kl_loss_ = th.mean(0.5 * th.sum(th.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        return kl_loss_ # scalar loss

    # def kl_loss(self, z_mu, z_logvar):
    #     kl_loss = -0.5 * th.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    #     return kl_loss # scalar loss

    def freeze(self):
        for module_ in self.encoder:
            for p in module_.parameters():
                p.requires_grad = False

        for module_ in self.mean:
            for p in module_.parameters():
                p.requires_grad = False

    # def forward(self, input):
    #     encoder = self.encoder(input)  # input: [batch_size,3,256,256], encoder: [1, 1024, 2, 2]
    #     z_mu = self.mean(encoder)   # [batch_size,1024,1,1]
    #     z_logvar = self.logvar(encoder)  # [batch_size,1024,1,1]
    #
    #     sample_z = self.sample_z(z_mu, z_logvar) # [batch_size,1024,1,1]
    #     kl_loss = self.kl_loss(z_mu, z_logvar) # scalar KL loss
    #     return sample_z, kl_loss, z_mu


    def forward(self, input):
        encoder = self.encoder(input)  # input: [1,3,200,200]
        z_mu = self.mean(encoder)  # [1,1024,1,1]
        sample_z = self.sample_z(z_mu) # [1,1024,1,1]
        kl_loss = self.kl_loss(z_mu) # scalar KL loss
        return sample_z, kl_loss, z_mu

    # def forward(self, input):
    #     encode128 = self.layer1(input)
    #     encode64 = self.layer2(encode128)
    #     encode32 = self.layer3(encode64)
    #     encode16 = self.layer4(encode32)
    #     encode8 = self.layer5(encode16)
    #     encode4 = self.layer6(encode8)
    #     encode2 = self.layer7(encode4)
    #     z_mu = self.mean(encode2)  # [1,1024,1,1]
    #     sample_z = self.sample_z(z_mu)  # [1,1024,1,1]
    #     return sample_z, z_mu, encode2, encode4, encode8, encode16, encode32, encode64, encode128
