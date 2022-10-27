import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import time
from torch import einsum
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in this work. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', unshared_layers=4, use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: da
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        unshared_layers (int) -- the number of unshared layers in the domain-invariant encoding-decoding stage
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    We adopt the UNet-like architecture of iDIH as the backbone of our generator [da].

    The generator has been initialized by <init_net>.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'da': #mia
        net = DAGenerator(input_nc, unshared_layers, ngf, norm_layer=norm_layer,batchnorm_from=2,image_fusion=True, attend_from=-1)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: fc
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    We adopt a fully-connected feature discriminator [fc].

    The discriminator has been initialized by <init_net>.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'fc':
        net = FCDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask, gp=True)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class FCDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf//2, 1, kernel_size=3, stride=1, padding=1)
            ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class StyleClassifier(nn.Module):
    def __init__(self, input_nc=128, ndf=64, n_class=10, norm_layer=nn.BatchNorm2d):
        super(StyleClassifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            norm_layer(ndf),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            norm_layer(ndf//2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ndf//2, ndf//4, kernel_size=3, stride=1, padding=1),
            norm_layer(ndf//4),
            nn.ReLU(True), 
            nn.MaxPool2d(4, 4),
            nn.Conv2d(ndf//4, n_class, kernel_size=3, stride=1, padding=1),
            )
    def forward(self, x):
        out = self.net(x)
        out = torch.squeeze(out)
        return F.softmax(out)



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,
        bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class ConvEncoder(nn.Module):
    def __init__(
        self,input_nc,depth,ngf,norm_layer,batchnorm_from):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        in_channels = input_nc
        out_channels = ngf

        self.block0 = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        for block_i in range(2, depth):
            # print(0 if block_i >= depth - 1 and out_channels == 512 else 1)
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, 512)

            self.blocks_connected[f'block{block_i}'] = ConvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding= 0 if block_i >= depth - 1 and out_channels == 512 else 1
            )
            self.blocks_channels += [out_channels]

    def forward(self, x):
        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            #print(f'unshared encoder {block_i}: {output.shape}')
            outputs += [block(output)]
        return outputs[::-1]

class AttDeconvBlock(nn.Module):
    def __init__(
        self,in_channels, out_channels,kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,with_att=False
    ):
        super(AttDeconvBlock, self).__init__()
        self.with_att = with_att
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.with_att:
            attention_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_sigmoid])

    def forward(self, x, mask=None):
        out = self.block(x)
        if self.with_att:
            out = self.attention(out)*out
        return out


class DeconvDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(AttDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 and out_channels == 256 else 1,
                with_att=0 <= attend_from <= d
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None, decoder_input=None):
        if decoder_input is None:
            output = encoder_outputs[0]
        else:
            output = decoder_input
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            #print(f'unshared decoder: {output.shape}, {skip_output.shape}')
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)
        return output,attention_map


class DAGenerator(nn.Module):
    def __init__(
        self,input_nc,unshared_depth,ngf,norm_layer=nn.BatchNorm2d,batchnorm_from=0,image_fusion=False, attend_from=-1):
        super(DAGenerator, self).__init__()
        self.generator_depth = 7
        self.unshared_depth = unshared_depth
        #unshared encoder for rendered domain
        self.render_enc = ConvEncoder(input_nc,unshared_depth,ngf,norm_layer,batchnorm_from)
        #unshared encoder for real domain
        self.real_enc = ConvEncoder(input_nc,unshared_depth,ngf,norm_layer,batchnorm_from)
        #shared encoder for both domains
        self.shared_blocks_channels = []
        self.encoder_blocks = nn.ModuleDict()
        out_channels = (2**(((self.unshared_depth-1))//2))*ngf
        for block_i in range(self.unshared_depth, self.generator_depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, 512)
            self.encoder_blocks[f'block{block_i}'] = ConvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=int(block_i < self.generator_depth - 1)
            )
            self.shared_blocks_channels += [out_channels]
        #shared decoder for both domains
        self.deconv_blocks = nn.ModuleList()
        in_channels = self.shared_blocks_channels.pop()
        out_channels = in_channels
        for d in range(self.generator_depth-self.unshared_depth):
            if len(self.shared_blocks_channels):
                out_channels = self.shared_blocks_channels.pop()  
            else:
                if unshared_depth%2==0:
                    out_channels = in_channels // 2
                else:
                    out_channels = in_channels
            self.deconv_blocks.append(AttDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_att=0 <= attend_from <= d
            ))
            in_channels = out_channels
        #unshared decoder for real domain
        self.real_dec = DeconvDecoder(unshared_depth, self.real_enc.blocks_channels, norm_layer, attend_from, image_fusion)
        #unshared decoder for rendered domain
        self.render_dec = DeconvDecoder(unshared_depth, self.render_enc.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, input, domain_flag=1):
        image = input[:,:3,:,:]
        mask = input[:,3,:,:]
        #unshared enc
        if domain_flag ==0: 
            unshared_enc_outputs = self.render_enc(input)
        else: 
            unshared_enc_outputs = self.real_enc(input)
        #shared enc-dec
        shared_enc_outputs = [unshared_enc_outputs[0]]
        for block_i in range(self.unshared_depth, self.generator_depth):
            block = self.encoder_blocks[f'block{block_i}']
            shared_enc_outputs += [block(shared_enc_outputs[-1])]
        shared_enc_outputs = shared_enc_outputs[::-1]
        output = shared_enc_outputs[0]
        for block, skip_output in zip(self.deconv_blocks, shared_enc_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        shared_dec_outputs = output
        #unshared dec
        if domain_flag ==0: 
            output, attention_map = self.render_dec(unshared_enc_outputs, image, mask, output)
        else:
            output, attention_map = self.real_dec(unshared_enc_outputs, image, mask, output)
        return unshared_enc_outputs[0],shared_dec_outputs, output, attention_map
