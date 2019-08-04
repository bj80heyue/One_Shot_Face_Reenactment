import torch as th
import torch.nn as nn


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = th.bmm(feat_t, feat) / (ch * h * w)
    gram = gram.view(b, h, w)
    return gram