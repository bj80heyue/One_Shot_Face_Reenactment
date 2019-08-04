# -*- coding: utf-8 -*-
import argparse
import torch


class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, misc_arg):
        # data set
        misc_arg.add_argument('--batch_size', type=int,
                            default=6, help='input batch size')
        misc_arg.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')

        # net set
        misc_arg.add_argument('--input_nc', type=int, default=9,
                            help='# of input image channels')
        misc_arg.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels')
        misc_arg.add_argument('--ngf', type=int, default=64,
                            help='# of gen filters in first conv layer')
        misc_arg.add_argument('--ndf', type=int, default=64,
                            help='# of discrim filters in first conv layer')

        misc_arg.add_argument('--netD', type=str, default='basic',
                            help='selects model to use for netD')

        misc_arg.add_argument('--n_layers_D', type=int, default=3,
                            help='only used if netD==n_layers')
        misc_arg.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # loss set
        misc_arg.add_argument('--loss_percept', action='store_true',
                              help='include perceptual loss')
        misc_arg.add_argument('--loss_faceID', action='store_true',
                              help='include face identity loss')
        # misc_arg.add_argument('--loss_percept', type=bool, default=True,
        #                       help='include perceptual loss')
        # misc_arg.add_argument('--loss_faceID', type=bool, default=True,
        #                       help='include face identity loss')

        misc_arg.add_argument('--gan_start_epoch', type=int, default=0,
                              help='start to include GAN loss from which epoch')

        # path and name
        misc_arg.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        misc_arg.add_argument('--load_path', type=str, default='trained_model')
        misc_arg.add_argument('--load_model_iter', type=str, default='latest',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        
        misc_arg.add_argument('--num_threads', default=4, type=int,
                            help='# threads for loading data')
        misc_arg.add_argument('--save_dir', type=str,
                            default='runs', help='output path')

        # norm and dropout
        misc_arg.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization for discriminator')
        # misc_arg.add_argument('--no_dropout', action='store_true',
        #                     help='no dropout for the generator')

        # init
        misc_arg.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        misc_arg.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        misc_arg.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        
        # display
        misc_arg.add_argument('--log_step', type=int, default=200,
                            help='log after n iters')
        misc_arg.add_argument('--save_step', type=int,
                            default=200, help='log after n iters')
                            
        misc_arg.add_argument('--save_by_iter', action='store_true',
                            help='whether saves model by iteration')
        misc_arg.add_argument('--phase', type=str, default='test',
                            help='train, val, test, etc')
        # optimizer                
        misc_arg.add_argument('--niter', type=int, default=100,
                            help='# of iter at starting learning rate')
        misc_arg.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')
        misc_arg.add_argument('--beta1', type=float, default=0.5,
                            help='momentum term of adam')
        misc_arg.add_argument('--lr', type=float, default=0.0001,
                            help='initial learning rate for adam')
        misc_arg.add_argument('--no_lsgan', action='store_true',
                            help='do *not* use least square GAN, if false, use vanilla GAN')
        misc_arg.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        
        misc_arg.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau|cosine')
        misc_arg.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        self.initialized = True
        return misc_arg

    def get_config(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        config, _ = parser.parse_known_args()

        # set gpu ids, transfrom string to int number
        str_ids = config.gpu_ids.split(',')
        config.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                config.gpu_ids.append(id)
        if len(config.gpu_ids) > 0:
            torch.cuda.set_device(config.gpu_ids[0])

        return config


    
