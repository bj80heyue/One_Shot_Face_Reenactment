import torch as th
from torch import nn
from net.ResNet import resnet_face18 as resnet18
from net.face_id_mlp_net import MLP
import net.base_net as base_net
###############################################################################
# define
###############################################################################


def defineFaceID(input_nc=3, class_num=10173, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    net = faceIDNet(input_nc, class_num)
    return base_net.init_net(net, init_type, init_gain, gpu_ids)


class faceIDNet(nn.Module):
    def __init__(self, input_nc, class_num):
        super(faceIDNet, self).__init__()
        # input 3x256x256
        self.feat = resnet18(input_nc, use_se=False)
        self.mlp = MLP(512, class_num)

    def forward(self, input):
        feat = self.feat(input)
        pred = self.mlp(feat)
        return pred

    def face_id_loss(self, x, target, loss_func):
        targetIdFeat256 = self.feat(target).detach()
        faceIDFeat = self.feat(x)
        id_loss = loss_func(faceIDFeat, targetIdFeat256)
        return id_loss
