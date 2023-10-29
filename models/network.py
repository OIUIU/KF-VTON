import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models
from options.train_options import TrainOptions
import os
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from models.external_function import simam

from collections import OrderedDict
opt = TrainOptions().parse()

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print("Network [{}] was created. Total number of parameters: {:.1f} million. "
              "To see the architecture, do print(network).".format(self.__class__.__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if 'BatchNorm2d' in classname:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif ('Conv' in classname or 'Linear' in classname) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method '{}' is not implemented".format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, *inputs):
        pass



class SPADEGenerator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super(SPADEGenerator, self).__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers

        # self.sh, self.sw = self.compute_latent_vector_size(opt)
        self.sh,self.sw = 256,192
        nf = opt.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 8, kernel_size=3, padding=1)
        for i in range(1, 5):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))

        self.head_0 = SPADEResBlock(opt, nf * 8, nf * 8, use_mask_norm=False)

        self.G_middle_0 = SPADEResBlock(opt, nf * 8 + 16, nf * 8, use_mask_norm=False)
        self.G_middle_1 = SPADEResBlock(opt, nf * 8 + 16, nf * 8, use_mask_norm=False)

        self.up_0 = SPADEResBlock(opt, nf * 8 + 16, nf * 4, use_mask_norm=False)
        self.up_1 = SPADEResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_2 = SPADEResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        # self.up_3 = SPADEResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = SPADEResBlock(opt, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def compute_latent_vector_size(self, opt):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 4
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))

        sh = opt.height // 2**num_up_layers
        sw = opt.width // 2**num_up_layers
        return sh, sw

    def forward(self, x, seg):
        samples_all = [F.interpolate(x, scale_factor=0.5 ** (3 - i), mode='nearest') for i in range(4)]
        # samples = [torch.cat((c[i], samples[i]), dim=1) for i in range(5)]
        features = [self._modules['conv_{}'.format(i)](samples_all[i]) for i in range(4)]



        x = self.head_0(features[0], seg)
        x = self.up(x)
        # x = self.G_middle_0(torch.cat((x, features[1]), 1), seg_2)
        # x = self.up(x)
        x = self.up_0(torch.cat((x, features[1]), 1), seg)
        # if self.num_upsampling_layers in ['more', 'most']:
        #     x = self.up(x)

        x = self.up(x)
        x = self.up_1(torch.cat((x, features[2]), 1), seg)

        x = self.up(x)
        x = self.up_2(torch.cat((x, features[3]), 1), seg)

        # x = self.up(x)
        # x = self.up_2(torch.cat((x, features[5]), 1), seg_2)
        # x = self.up(x)
        # x = self.up_3(torch.cat((x, features[6]), 1), seg_2)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)
class SPADEResBlock(nn.Module):
    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        super(SPADEResBlock, self).__init__()

        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(output_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(output_nc, output_nc, kernel_size=1, bias=False)

        subnorm_type = opt.norm_G
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        gen_semantic_nc = opt.gen_semantic_nc
        if use_mask_norm:
            subnorm_type = 'aliasmask'
            gen_semantic_nc = gen_semantic_nc + 1

        self.norm_0 = SPADENorm(input_nc,output_nc, gen_semantic_nc,subnorm_type)
        self.norm_1 = SPADENorm(output_nc,output_nc, gen_semantic_nc,subnorm_type)
        if self.learned_shortcut:
            self.norm_s = SPADENorm(input_nc,output_nc, gen_semantic_nc,subnorm_type)

        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg, misalign_mask):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, misalign_mask))
        else:
            return x

    def forward(self, x, seg, misalign_mask=None):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        x_s = self.shortcut(x, seg, misalign_mask)
        dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        output = simam(x_s + dx)
        return output



class SPADENorm(nn.Module):
    def __init__(self,input_nc, out_nc, label_nc, norm_type):
        super(SPADENorm, self).__init__()

        self.noise_scale = nn.Parameter(torch.zeros(input_nc))

        assert norm_type.startswith('alias')
        param_free_norm_type = norm_type[len('alias'):]
        if param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(out_nc, affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(out_nc, affine=False)
        elif param_free_norm_type == 'mask':
            self.param_free_norm = MaskNorm(out_nc)
        else:
            raise ValueError(
                "'{}' is not a recognized parameter-free normalization type in SPADENorm".format(param_free_norm_type)
            )
        self.conv2d = nn.Conv2d(input_nc, out_nc, 3, 1, padding=1)
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.conv_shared = nn.Sequential(
            nn.Conv2d(out_nc, nhidden, kernel_size=ks, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(nhidden, out_nc, kernel_size=ks, padding=1)
        self.conv_beta = nn.Conv2d(nhidden, out_nc, kernel_size=ks, padding=1)

        self.conv_shared_2 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=1),
            nn.ReLU()
        )
        self.gamma_seg_gamma = nn.Conv2d(nhidden, out_nc, kernel_size=3, padding=1)
        self.beta_seg_gamma = nn.Conv2d(nhidden, out_nc, kernel_size=3, padding=1)

        self.gamma_seg_beta = nn.Conv2d(nhidden, out_nc, kernel_size=3, padding=1)
        self.beta_seg_beta = nn.Conv2d(nhidden, out_nc, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x, seg, misalign_mask=None):
        # Part 1. Generate parameter-free normalized activations.
        b, c, h, w = x.size()

        # normalized = self.param_free_norm(x)
        x = self.conv2d(x)
        # noise = (torch.randn(b, w, h, 1).cuda() * self.noise_scale).transpose(1, 3)

        if misalign_mask is None:
            normalized = self.param_free_norm(x)
        else:
            normalized = self.param_free_norm(x, misalign_mask)
        seg = F.interpolate(seg, size=normalized.size()[2:], mode='nearest')
        x = F.interpolate(x, size=normalized.size()[2:], mode='nearest')
        # Part 2. Produce affine parameters conditioned on the segmentation map.
        # actv = self.conv_shared(seg_2)
        # gamma = self.conv_gamma(actv)
        # beta = self.conv_beta(actv)
        seg = self.conv_shared_2(seg)
        gamma_seg_gamma = self.gamma_seg_gamma(seg)
        beta_seg_gamma = self.beta_seg_gamma(seg)
        gamma_seg_beta = self.gamma_seg_beta(seg)
        beta_seg_beta = self.beta_seg_beta(seg)

        actv = self.conv_shared(x)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        gamma = gamma * (1. + gamma_seg_gamma) + beta_seg_gamma
        beta = beta * (1. + gamma_seg_beta) + beta_seg_beta
        out_norm = normalized * (1. + gamma) + beta

        output = self.activation(out_norm)
        # Apply the affine parameters.
        # output = normalized * (1 + gamma) + beta
        return output

class MaskNorm(nn.Module):
    def __init__(self, norm_nc):
        super(MaskNorm, self).__init__()

        self.norm_layer = nn.InstanceNorm2d(norm_nc, affine=False)

    def normalize_region(self, region, mask):
        b, c, h, w = region.size()

        num_pixels = mask.sum((2, 3), keepdim=True)  # size: (b, 1, 1, 1)
        num_pixels[num_pixels == 0] = 1
        mu = region.sum((2, 3), keepdim=True) / num_pixels  # size: (b, c, 1, 1)

        normalized_region = self.norm_layer(region + (1 - mask) * mu)
        return normalized_region * torch.sqrt(num_pixels / (h * w))

    def forward(self, x, mask):
        mask = mask.detach()
        normalized_foreground = self.normalize_region(x * mask, mask)
        normalized_background = self.normalize_region(x * (1 - mask), 1 - mask)
        return normalized_foreground + normalized_background

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.old_lr = opt.lr
        self.old_lr_gmm = 0.1*opt.lr

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# def save_checkpoint(model, save_path):
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     torch.save(model.state_dict(), save_path)
def save_checkpoint(model,optimizer,epoch,lr_schedule, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        'lr_schedule': lr_schedule.state_dict()
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('----No checkpoints at given path----')
        return

    state_dict =  torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.cuda()
    print('----checkpoints loaded from path: {}----'.format(checkpoint_path))

def load_checkpoint1(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('----No checkpoints at given path----')
        return

    state_dict =  torch.load(checkpoint_path)
    model.load_state_dict(state_dict['net'])
    model.cuda()
    print('----checkpoints loaded from path: {}----'.format(checkpoint_path))

def load_checkpoint_parallel(model,checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(opt.local_rank))
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)

def load_checkpoint_part_parallel(model,checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return
    checkpoint = torch.load(checkpoint_path,map_location='cuda:{}'.format(opt.local_rank))
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        if 'cond_' not in param and 'aflow_net.netRefine' not in param:
            checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)


