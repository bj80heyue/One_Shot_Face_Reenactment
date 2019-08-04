import os
import torch
from collections import OrderedDict
import net.base_net as base_net
import shutil
from tensorboardX import SummaryWriter
import json
import random
import logging
import datetime


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = ('train' == opt.phase)
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # self.save_dir = opt.save_dir
        self.loss_names = []
        self.visual_names = []
        self.image_paths = []
        self.train_model_name = []

        # if os.path.exists(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        # os.makedirs(self.save_dir)

    def set_input(self, input):
        pass

    def forward(self):
        pass

    def set_logger(self, opt):
        if self.isTrain:
            run_id = random.randint(1,100000)
            self.logdir = os.path.join(opt.save_dir,str(run_id))
            self.writer = SummaryWriter(self.logdir)
            self.logger = self.get_logger(self.logdir)
            self.logger.info('Let the games begin')
            self.logger.info('save dir: runs/{}'.format(run_id))
            print('log dir : ', self.logdir)
        else:
            self.logdir = os.path.join(opt.save_dir,'test_res')
            self.writer = SummaryWriter(self.logdir)

    def get_logger(self, logdir):
        logger = logging.getLogger('myLogger')
        ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
        ts = ts.replace(":", "_").replace("-", "_")
        file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
        hdlr = logging.FileHandler(file_path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        return logger

    def save_config(self, config):
        param_path = os.path.join(self.logdir, "params.json")
        print("[*] PARAM path: %s" % param_path)

        with open(param_path, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)


    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [base_net.get_scheduler(
                optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.load_path:
            # load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            load_suffix = '{}/{}_net'.format(opt.load_path, opt.load_model_iter)
            self.load_networks_all(load_suffix)
            print('load {} successful!'.format(load_suffix))
        self.print_networks(opt.verbose)

    def load_networks_all(self, prefix):
        for name in self.train_model_name:
            if 'netD' in name:
                continue
            net = getattr(self, name)
            load_filename = '{}_{}.pth'.format(prefix, name)

            self.load_networks(net, load_filename)

    # load model
    def load_networks(self, model, path):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        pretrainDict = torch.load(path, map_location=self.device)
        modelDict = model.state_dict()
        for kk, vv in pretrainDict.items():
            kk = kk.replace('module.', '')
            if kk in modelDict:
                modelDict[kk].copy_(vv)
            else:
                print('{} not in modelDict'.format(kk))
        # model.load_state_dict(pretrainDict)
        # print(modelDict.keys())

    # make models eval mode during test time
    def eval(self):
        for name in self.train_model_name:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # # return visualization images. train.py will display these images, and save the images to a html
    # def get_current_visuals(self):
    #     visual_ret = OrderedDict()
    #     for name in self.visual_names:
    #         if isinstance(name, str):
    #             visual_ret[name] = getattr(self, name)
    #     return visual_ret
    #
    # # return traning losses/errors. train.py will print out these errors as debugging information
    # def get_current_losses(self):
    #     errors_ret = OrderedDict()
    #     for name in self.loss_names:
    #         if isinstance(name, str):
    #             # float(...) works for both scalar tensor and float number
    #             errors_ret[name] = float(getattr(self, 'loss_' + name))
    #     return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.train_model_name:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.logdir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                    if len(self.gpu_ids) > 1:
                        net = torch.nn.DataParallel(net, self.opt.gpu_ids)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # print network information

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.train_model_name:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
