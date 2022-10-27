import torch
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import functools
from torch.nn.utils import spectral_norm
from util.losses import MSE, MaskWeightedMSE, SoftCrossEntropy, CrossEntropy


class DAstyleModel(BaseModel): #unet model with foreground-normed mse loss
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_src_L1', type=float, default=10.0, help='weight for L1 loss in rendered domain')
            parser.add_argument('--lambda_tgt_L1', type=float, default=10.0, help='weight for L1 loss in real domain')
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--lambda_m', type=float, default=1.0, help='threshold for lowerEntropy loss in real domain') #added by Joy
            parser.add_argument('--lambda_tgt_le', type=float, default=0.5, help='weight for loss_tgt_lowerEntropy loss in real domain') #added by mia
            parser.add_argument('--lambda_tgt_ce', type=float, default=0.5, help='weight for loss_tgt_crossEntropy loss in real domain') #added by mia
            parser.add_argument('--lambda_src_ce', type=float, default=1.0, help='weight for the SoftCrossEntropy and CrossEntropy loss in rendered domain') #added by mia

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_src_L1','G_tgt_L1','G_GAN','D_src','D_tgt','D_loss',\
            'tgt_crossEntropy','tgt_lowerEntropy','style_loss','src_before_harm', 'src_after_harm']
        
        if self.isTrain:
            self.visual_names = ['src_comp','src_mask', 'src_real', 'src_harm','src_att_map',\
                            'tgt_comp','tgt_mask', 'tgt_real', 'tgt_harm','tgt_att_map']
            self.model_names = ['G', 'D', 'SC1', 'SC2']
        else: 
            self.visual_names = ['tgt_comp','tgt_mask', 'tgt_real', 'tgt_harm','tgt_att_map']
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, opt.unshared_layers,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:
            self.netD = networks.define_D((2**(((opt.unshared_layers-1))//2))*opt.ndf, opt.ndf, opt.netD, opt.n_layers_D,
                                    opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            netSC1 = networks.StyleClassifier((2**(((opt.unshared_layers-1))//2))*opt.ndf, opt.ndf, 10)
            self.netSC1 = networks.init_net(netSC1, opt.init_type, opt.init_gain, self.gpu_ids)
            netSC2 = networks.StyleClassifier((2**(((opt.unshared_layers-1))//2))*opt.ndf, opt.ndf, 10)
            self.netSC2 = networks.init_net(netSC2, opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            self.gan_mode = opt.gan_mode
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionfnmse = MaskWeightedMSE(min_area=100)
            self.SoftCrossEntropy = SoftCrossEntropy()
            self.crossEntropy = CrossEntropy() # added by Joy
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*opt.d_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers_G = torch.optim.Adam([ {'params': self.netG.parameters(), 'lr': opt.lr*opt.g_lr_ratio, 'betas': (opt.beta1, 0.999)}, \
                {'params': self.netSC2.parameters(), 'lr': opt.lr*opt.sc_lr_ratio, 'betas': (opt.beta1, 0.999)},\
                {'params': self.netSC1.parameters(), 'lr': opt.lr*opt.sc_lr_ratio, 'betas': (opt.beta1, 0.999)} ])
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizers_G)

    def set_input(self, tgt_input, src_input=None):
        if self.isTrain:
            self.src_comp = src_input['comp'].to(self.device)
            self.src_mask = src_input['mask'].to(self.device)
            self.src_real = src_input['real'].to(self.device)
            self.src_inputs = torch.cat([src_input['comp'],src_input['mask']],1).to(self.device)
            self.src_before_harm_gt = src_input['unharm_GT_label'].to(self.device)
            self.src_after_harm_gt = src_input['harmed_GT_label'].to(self.device)
        self.tgt_comp = tgt_input['comp'].to(self.device)
        self.tgt_mask = tgt_input['mask'].to(self.device)
        self.tgt_real = tgt_input['real'].to(self.device)
        self.tgt_inputs = torch.cat([tgt_input['comp'],tgt_input['mask']],1).to(self.device)
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.src_fea, self.src_harm_fea, self.src_harm, self.src_att_map = self.netG(self.src_inputs, 0)  # G(A)
        self.tgt_fea, self.tgt_harm_fea, self.tgt_harm, self.tgt_att_map = self.netG(self.tgt_inputs, 1)  # G(A)


    def backward_D(self):
        #rendered domain
        fake_fea = self.src_fea
        pred_fake= self.netD(fake_fea.detach())
        if self.gan_mode == 'wgangp':
            loss_D_src = self.relu(1 + pred_fake).mean()
        else:
            loss_D_src = self.criterionGAN(pred_fake, False)
        self.loss_D_src = loss_D_src

        #real domain
        real_fea = self.tgt_fea
        pred_real = self.netD(real_fea.detach())
        if self.gan_mode == 'wgangp':
            loss_D_tgt = self.relu(1 - pred_real).mean()
        else:
            loss_D_tgt = self.criterionGAN(pred_real, True)
        self.loss_D_tgt = loss_D_tgt

        self.loss_D_loss = self.loss_D_src + self.loss_D_tgt
        self.loss_D_loss.backward(retain_graph=True) #added by mia

    def backward_G(self):
        #gan loss
        fake_fea = self.src_fea
        pred_fake = self.netD(fake_fea)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        #style in rendered domain. Joy
        fake_harm_fea = self.src_harm_fea
        before_harm_fake = self.netSC1(fake_fea)
        after_harm_fake = self.netSC2(fake_harm_fea)
        self.loss_src_before_harm = self.SoftCrossEntropy(before_harm_fake, self.src_before_harm_gt)
        self.loss_src_after_harm = self.crossEntropy(after_harm_fake, self.src_after_harm_gt)

        #style in real domain
        real_fea = self.tgt_fea
        real_harm_fea = self.tgt_harm_fea
        before_harm_real = self.netSC1(real_fea)
        after_harm_real = self.netSC2(real_harm_fea)

        self.loss_tgt_crossEntropy = self.SoftCrossEntropy(after_harm_real, before_harm_real)
        self.loss_tgt_lowerEntropy = max(0, self.opt.lambda_m - self.SoftCrossEntropy(before_harm_real, before_harm_real) + self.SoftCrossEntropy(after_harm_real, after_harm_real))

        self.loss_style_loss = (self.loss_src_before_harm + self.loss_src_after_harm)*self.opt.lambda_src_ce + \
                    self.loss_tgt_crossEntropy*self.opt.lambda_tgt_ce + self.loss_tgt_lowerEntropy*self.opt.lambda_tgt_le
        
        self.loss_G_src_L1 = torch.mean(self.criterionfnmse(self.src_harm, self.src_real, self.src_mask) * self.opt.lambda_src_L1)
        self.loss_G_tgt_L1 = torch.mean(self.criterionfnmse(self.tgt_harm, self.tgt_real, self.tgt_mask) * self.opt.lambda_tgt_L1)
        self.loss_G = self.loss_G_GAN*self.opt.lambda_adv + self.loss_G_src_L1 + self.loss_G_tgt_L1 + self.loss_style_loss
        self.loss_G.backward(retain_graph=True) 

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizers_G.zero_grad()
        self.backward_G()
        self.optimizers_G.step()
