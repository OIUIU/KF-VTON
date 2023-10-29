import torch.nn.functional as F
import torch
import torch.nn as nn
from models.network import BaseNetwork,SPADEGenerator
from options.train_options import TrainOptions
from models.keypoint_detector import KPDetector
from .dense_motion import DenseMotionNetwork
import numpy as np
from torchvision import transforms
from models.util import ResBlock2d
from PIL import Image, ImageDraw
from models.external_function import ResBlock
class WarpNetwork(BaseNetwork):
    def __init__(self, opt, block_expansion,max_features,gen_input):
        super(WarpNetwork, self).__init__()
        self.opt = opt
        num_tps = opt.num_tps
        num_kp = opt.num_kp
        if opt.phase == 'train':
            self.old_lr = opt.lr_warp
        self.kp_detector = KPDetector(num_tps=num_tps,num_kp = num_kp,num_channels=3,size=32*24)

        self.self_dense_motion_network = DenseMotionNetwork(opt,block_expansion=block_expansion, num_blocks=4, max_features=max_features,
                                                            num_tps=num_tps,num_kp = num_kp)

        self.generator = SPADEGenerator(opt, gen_input)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, mode='bilinear', padding_mode='border')

    def occlude_input(self, inp, occlusion_map):

        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out


    def forward(self,image_input,person_clothes, pose, img_agnostic,ref_input,gen_parse,dropout_flag, dropout_p):

        kp_source, kp_ref= self.kp_detector(image_input,ref_input)
        dense_motion = self.self_dense_motion_network(image_input, ref_input,kp_source,kp_ref ,dropout_flag, dropout_p)
        deformation = dense_motion['deformation']
        occlusion_map = dense_motion['occlusion_map']
        deformed_source = self.deform_input(image_input, deformation)
        warped_out = self.occlude_input(deformed_source, occlusion_map)

        tryon = self.generator(torch.cat((img_agnostic,pose, warped_out), dim=1), gen_parse)
        return tryon,warped_out,kp_source, kp_ref,dense_motion

    def update_learning_rate(self,optimizer):
        lrd = self.opt.lr_warp / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

