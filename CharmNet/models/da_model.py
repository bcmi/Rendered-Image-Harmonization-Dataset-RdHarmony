import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import functools
from torch.nn.utils import spectral_norm
from util.losses import MSE, MaskWeightedMSE


class DAModel(BaseModel): #unet model with foreground-normed mse loss
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_src_L1', type=float, default=100.0, help='weight for L1 loss in rendered domain')
            parser.add_argument('--lambda_tgt_L1', type=float, default=100.0, help='weight for L1 loss in real domain')
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_src_L1','G_tgt_L1','G_GAN','D_src','D_tgt','D_loss']
        
        if self.isTrain:
            self.visual_names = ['src_comp','src_mask', 'src_real', 'src_harm','src_att_map',\
                            'tgt_comp','tgt_mask', 'tgt_real', 'tgt_harm','tgt_att_map']
            self.model_names = ['G', 'D']
        else: 
            self.visual_names = ['tgt_comp','tgt_mask', 'tgt_real', 'tgt_harm','tgt_att_map']
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, opt.unshared_layers,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:
            self.netD = networks.define_D((2**(((opt.unshared_layers-1))//2))*opt.ndf, opt.ndf, opt.netD, opt.n_layers_D,
                                    opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            self.gan_mode = opt.gan_mode
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionfnmse = MaskWeightedMSE(min_area=100)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*opt.d_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    def set_input(self, tgt_input, src_input=None):
        if self.isTrain:
            self.src_comp = src_input['comp'].to(self.device)
            self.src_mask = src_input['mask'].to(self.device)
            self.src_real = src_input['real'].to(self.device)
            self.src_inputs = torch.cat([src_input['comp'],src_input['mask']],1).to(self.device)
        self.tgt_comp = tgt_input['comp'].to(self.device)
        self.tgt_mask = tgt_input['mask'].to(self.device)
        self.tgt_real = tgt_input['real'].to(self.device)
        self.tgt_inputs = torch.cat([tgt_input['comp'],tgt_input['mask']],1).to(self.device)
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.src_fea,_, self.src_harm, self.src_att_map = self.netG(self.src_inputs, 0)  # G(A)
        self.tgt_fea,_, self.tgt_harm, self.tgt_att_map = self.netG(self.tgt_inputs, 1)  # G(A)
    
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
        fake_fea = self.src_fea
        pred_fake = self.netD(fake_fea)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_src_L1 = torch.mean(self.criterionfnmse(self.src_harm, self.src_real, self.src_mask) * self.opt.lambda_src_L1)
        self.loss_G_tgt_L1 = torch.mean(self.criterionfnmse(self.tgt_harm, self.tgt_real, self.tgt_mask) * self.opt.lambda_tgt_L1)
        self.loss_G = self.loss_G_GAN*self.opt.lambda_adv + self.loss_G_src_L1 + self.loss_G_tgt_L1
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
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

