import torch as th
from torch import nn
from torchvision.models import vgg16

import net.base_net as base_net
from utils.metric import gram_matrix
###############################################################################
# define
###############################################################################


def defineVGG(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = VGGNet()
    return base_net.init_net(net, init_type, init_gain, gpu_ids)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.net = vgg16()
        vgg_path = 'pretrainModel/vgg16-397923af.pth'
        self.net.load_state_dict(th.load(vgg_path))

    def forward(self, x):
        map_ = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        vgg_layers = self.net.features
        layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        output = []
        for name, module in vgg_layers._modules.items():
            x = module(x)
            #x = nn.parallel.data_parallel(module, x, range(num_gpu))
            if name in layer_name_mapping:
                output.append(x)
        return output

    def perceptual_loss(self, x, target, loss_func):
        self.x_result = self.forward(x)
        self.target_result = self.forward(target)
        loss_ = 0
        for xx, yy in zip(self.x_result, self.target_result):
            loss_ += loss_func(xx, yy.detach())
        return loss_

    def style_loss(self, x, target, loss_func):
        #x_result = self.forward(x)
        #target_result = self.forward(target)
        loss_ = 0
        for xx, yy in zip(self.x_result, self.target_result):
            loss_ += loss_func(gram_matrix(xx), gram_matrix(yy.detach()))
        return loss_