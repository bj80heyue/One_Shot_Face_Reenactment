import sys
from model.base_model import BaseModel
import net.vgg_net as vgg_net
import net.generaotr_net as generator_net
import net.discriminator_net as discriminator_net
import net.appear_decoder_net as appDec
import net.appear_encoder_net as appEnc
import net.face_id_net as face_id_net
import torch
import torch.nn.functional as F
import itertools
from utils import metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpadeModel(BaseModel):
    def __init__(self, opt):
        super(SpadeModel, self).initialize(opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['vgg', 'id', 'reconstruct', 'gan']
        self.train_model_name = ['appEnc', 'appDnc', 'netG']
        # define appearance encoder, decoder
        self.appEnc = appEnc.defineAppEnc(
            3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=self.opt.gpu_ids, conv_k=3)
        self.appDnc = appDec.defineAppDec(
            3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=self.opt.gpu_ids)
        self.netG = generator_net.defineSPADEGenerator(opt.input_nc, opt.output_nc, 64, norm='instance',
                                                       init_type='normal', init_gain=0.02, gpu_ids=self.opt.gpu_ids,
                                                       latent_chl=1024, up_mode='convT')
        # -----pass-----
        if self.isTrain:
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
            self.pretrain_model_name = []
            if self.opt.loss_percept:
                self.pretrain_model_name.append('vgg')
            if self.opt.loss_faceID:
                self.pretrain_model_name.append('faceId')

            self.train_model_name += ['netD256', 'netD128', 'netD64']
            # load vgg and faceID networks
            if self.opt.loss_percept:
                self.vgg = vgg_net.defineVGG(
                    init_type='no', gpu_ids=self.opt.gpu_ids).eval()
            if self.opt.loss_faceID:
                self.faceId = face_id_net.defineFaceID(
                    input_nc=opt.output_nc, gpu_ids=self.opt.gpu_ids).eval()
                faceId_path = 'pretrainModel/id_200.pth'
                self.load_networks(self.faceId, faceId_path)

            use_sigmoid = opt.no_lsgan
            self.netD256 = discriminator_net.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type,
                                                      opt.init_gain,
                                                      self.gpu_ids)
            self.netD128 = discriminator_net.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                      2, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                      self.gpu_ids)
            self.netD64 = discriminator_net.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                     2, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                     self.gpu_ids)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.netG.parameters(),
                                                                       self.appEnc.parameters(),
                                                                       self.appDnc.parameters())),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD256.parameters(), self.netD128.parameters(), self.netD64.parameters()),
                lr=0.5 * opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.loss_D = torch.tensor(0).float().to(
                device)  # initialize D_loss and gan_loss for G to 0 (first several epochs may not use gan)
            self.loss_gan = torch.tensor(0).float().to(device)

            # define loss functions
            self.criterionVGG = torch.nn.L1Loss().to(self.device)
            self.criterionId = torch.nn.L1Loss().to(self.device)
            self.criterionReconstruct = torch.nn.L1Loss().to(self.device)
            self.criterionPix = torch.nn.L1Loss().to(self.device)
            self.criterionGAN = discriminator_net.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)


    def set_input(self, input):
        self.seg_dst = input['seg_dst'].to(self.device)
        self.img_src = input['img_src'].to(self.device)
        self.srcMask = input['face_mask_src'].to(self.device)
        self.dstMask = input['face_mask_dst'].to(self.device)
        # apply ref & mask
        if 'img_dst' in input and self.isTrain:
            self.groundtruth = input['img_dst'].to(self.device)
            self.groundtruth = self.groundtruth * self.srcMask
        if 'weighted_mask_dst' in input:
            self.weightMask = input['weighted_mask_dst'].to(self.device)


    def forward(self):
        sample_z, kl_loss, _ = self.appEnc(self.img_src)  # [batch_size,1024,1,1]
        out16, out32, out64, out128, self.out256 = self.appDnc(sample_z)  # [1024, 16, 16,] [512, 32, 32], [256, 64, 64], [128, 128, 128], [3, 256, 256]
        self.fake_B = self.netG(self.seg_dst, sample_z, [out16, out32, out64, out128]) # [batch_size, 3, 256, 256]

        if self.isTrain:
            self.gt128 = F.max_pool2d(self.groundtruth, 3, stride=2)
            self.gt64 = F.max_pool2d(self.gt128, 3, stride=2)
            self.fake128 = F.max_pool2d(self.fake_B, 3, stride=2)
            self.fake64 = F.max_pool2d(self.fake128, 3, stride=2)
        return self.fake_B

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        return self.loss_D

    def backward_D(self):
        self.lossD256 = self.backward_D_basic(self.netD256, self.groundtruth, self.fake_B)
        self.lossD128 = self.backward_D_basic(self.netD128, self.gt128, self.fake128)
        self.lossD64 = self.backward_D_basic(self.netD64, self.gt64, self.fake64)
        self.loss_D = self.lossD256 + self.lossD128 + self.lossD64
        self.loss_D.backward()

    def backward_G(self, epoch):
        # perceptual loss
        if self.opt.loss_percept:
            self.loss_vgg = self.opt.lambda_vgg * (
                self.vgg.module.perceptual_loss(self.fake_B, self.groundtruth, self.criterionVGG) if hasattr(
                    self.vgg, 'module') else self.vgg.perceptual_loss(self.fake_B, self.groundtruth, self.criterionVGG))

        # Identity loss
        if self.opt.loss_faceID:
            fake_B_id = self.fake_B
            gt_id = self.groundtruth
            fake_B_id = fake_B_id[:,:,28:228, 28:228]
            gt_id = gt_id[:,:,28:228, 28:228]
            self.loss_id = self.opt.lambda_id * (
                self.faceId.module.face_id_loss(fake_B_id, gt_id, self.criterionId) if hasattr(
                    self.faceId, 'module') else self.faceId.face_id_loss(fake_B_id, gt_id, self.criterionId))

        # GAN loss
        if epoch >= self.opt.gan_start_epoch:
                self.loss_gan = self.opt.lambda_gan * (
                    self.criterionGAN(self.netD256(self.fake_B), True) + self.criterionGAN(self.netD128(self.fake128), True) + self.criterionGAN(self.netD64(self.fake64), True))

        # AE reconstruction loss
        self.loss_reconstruct = self.opt.lambda_reconstruct * self.criterionReconstruct(self.out256, self.img_src)

        # pixel loss between gt and generated image
        fake_B_pix = self.fake_B * (0.5 + self.weightMask)
        gt_pix = self.groundtruth * (0.5 + self.weightMask)
        self.loss_pix = self.opt.lambda_pix * self.criterionPix(fake_B_pix, gt_pix)

        # combined loss
        self.loss_G = torch.tensor(0).float().to(device)
        self.loss_G += self.loss_reconstruct
        self.loss_G += self.loss_pix
        if self.opt.loss_percept:
            self.loss_G += self.loss_vgg
        if self.opt.loss_faceID:
            self.loss_G += self.loss_id
        self.loss_G += self.loss_gan

        self.loss_G.backward()


    def func_require_grad(self, model_, flag_):
        for mm in model_:
            self.set_requires_grad(mm, flag_)

    def func_zero_grad(self, model_):
        for mm in model_:
            mm.zero_grad()

    def optimize_parameters(self, epoch):
        self.forward()
        # D
        if epoch >= self.opt.gan_start_epoch:  # start to include D after xxx epochs
            self.func_require_grad([self.netD256, self.netD128, self.netD64], True)
            self.func_zero_grad([self.netD256, self.netD128, self.netD64])
            self.backward_D()
            self.optimizer_D.step()
        # G
        self.func_require_grad([self.netD256, self.netD128, self.netD64], False)
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)  # start to include gan loss for G after xxx epochs
        self.optimizer_G.step()
