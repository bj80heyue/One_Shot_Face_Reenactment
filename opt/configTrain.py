# -*- coding: utf-8 -*-
from opt.config import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, misc_arg):
        misc_arg = BaseOptions.initialize(self, misc_arg)
        misc_arg.add_argument('--lambda_vgg', type=int, default=1)
        misc_arg.add_argument('--lambda_reconstruct', type=int, default=25)
        misc_arg.add_argument('--lambda_pix', type=int, default=25)
        misc_arg.add_argument('--lambda_id', type=int, default=1)
        misc_arg.add_argument('--lambda_gan', type=int, default=1)
        self.initialized = True
        return misc_arg

