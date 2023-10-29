from torch import nn
import torch.nn.functional as F
import torch
from .util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from .util import to_homogeneous, from_homogeneous,UpBlock2d,TPS
import math
from options.base_options import BaseOptions
from torchvision.transforms import ToTensor, ToPILImage
import cv2
import numpy as np

from torchvision.transforms import Resize
class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self,opt, block_expansion, num_blocks, max_features, num_tps,num_kp,multi_mask=False, num_channels=3,
                 scale_factor=0.5, bg = False,  kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.opt = opt
        self.num_tps = num_tps
        self.num_kp = num_kp

        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels *2* num_tps + num_tps*num_kp),
                                max_features=max_features, num_blocks=num_blocks)
        self.mask = nn.Conv2d(self.hourglass.out_filters , num_tps, kernel_size=(7, 7), padding=(3, 3))

        # for testing without bg motion
        self.source_features = FeatureEncoder(3)
        self.reference_features = FeatureEncoder(3)
        self.Cross_MFEs = MFEBlock(in_channels=256, out_channels=num_tps)

        self.bg = bg
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)


        if multi_mask:
            up = []
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 5

            channel = [self.hourglass.out_channels[- 1] // (2 ** i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [self.hourglass.out_channels[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(self.hourglass.out_channels[-1] // (2 ** (i + 1)))
            occlusion = []

            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            self.occlusion = nn.Conv2d(self.hourglass.out_channels[-1], 1, kernel_size=(7, 7), padding=(3, 3))



    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source, bg_params=None):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        # K TPS transformaions

        bs, _, h, w = source_image.shape
        kp_1 = kp_driving
        kp_2 = kp_source
        kp_1 = kp_1.view(bs, -1, self.num_kp, 2)#b,10,6,2
        kp_2 = kp_2.view(bs, -1, self.num_kp, 2)
        trans = TPS(self.opt,mode='kp', bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source_image)

        sparse_motions = driving_to_source
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps, 1, 1, 1, 1)#b,10,1,64,48,2
        source_repeat = source_repeat.view(bs * (self.num_tps ), -1, h, w)#10b,256,64,48
        sparse_motions = sparse_motions.view((bs * (self.num_tps ), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
        sparse_deformed = sparse_deformed.view((bs, self.num_tps, -1, h, w))
        return sparse_deformed


    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''
        drop = (torch.rand(X.shape[0],X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2],X.shape[3],1,1).permute(2,3,0,1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:,1:,...] /= (1-P)
        mask_bool =(drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition


    def forward(self, source_image,ref_input ,kp_source,kp_driving, dropout_flag=False, dropout_p=0, bg_params=None):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            ref_input = self.down(ref_input)

        all_image = torch.cat([source_image,ref_input],1)
        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)#b,60,64,48
        sparse_motion = self.create_sparse_motions(all_image, kp_driving, kp_source, bg_params=None)#b,10,64,48,2
        deformed_source = self.create_deformed_source_image(all_image, sparse_motion)#B,10,6,H,W

        out_dict['sparse_deformed'] = deformed_source
        deformed_source = deformed_source.view(bs, -1, h, w) #b,60,64,48
        input = torch.cat([heatmap_representation, deformed_source], dim=1)#B,120,H,W
        input = input.view(bs, -1, h, w)


        prediction = self.hourglass(input, mode = 1)

        contribution_maps = self.mask(prediction[-1])

        if (dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)

        # mask = F.softmax(mask, dim=1)
        out_dict['contribution_maps'] = contribution_maps


        contribution_maps = contribution_maps.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)#B,10,2,H,W
        # n * 11 * 2 * h * w
        deformation = (sparse_motion * contribution_maps).sum(dim=1)#B,2,H,W
        deformation = deformation.permute(0, 2, 3, 1)#B,H,W,2
        # n * h * w * 2

        out_dict['deformation'] = deformation
        out_dict['sparse_motion'] = sparse_motion
        out_dict['prediction'] = prediction

        occlusion_map = torch.sigmoid(self.occlusion(prediction[-1]))
        occlusion_map = F.interpolate(occlusion_map,size=[256,192], mode='bilinear',align_corners=True)

        out_dict['occlusion_map'] = occlusion_map
        return out_dict




def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)

class MFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_filters=[256,128,64]):
        super(MFEBlock, self).__init__()
        layers = []
        for i in range(len(num_filters)):
            if i==0:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=num_filters[i], kernel_size=3, stride=1, padding=1))
            else:
                layers.append(torch.nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            layers.append(torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
        layers.append(torch.nn.Conv2d(in_channels=num_filters[-1], out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        # print(self.layers)
        return self.layers(input)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64,128]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i - 1], out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block=  nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x) + x
