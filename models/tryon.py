import sys

import torch

import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from models.base_model import BaseModel
from options.train_options import TrainOptions
from util.util import gen_noise, get_palette
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchgeometry as tgm
from PIL import Image, ImageDraw


class TryOnNetwork(torch.nn.Module):
    def name(self):
        return 'TryOn'

    def __init__(self, opt,seg_model, warp_model):
        super(TryOnNetwork, self).__init__()
        # BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        self.up = nn.Upsample(size=(opt.height, opt.width), mode='bilinear')
        self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
        self.gauss.cuda()
        self.seg = seg_model
        # input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.sigmoid = nn.Sigmoid()

        self.warp = warp_model

    def forward(self,opt, image, img_agnostic, parse_agnostic, parse_all, pose,c_masked,
                dropout_flag,dropout_p,step):

        unloader = transforms.ToPILImage()

        # Part 1. Segmentation generation
        labels8 = {
            0: ['background', [0]],
            1: ['paste', [1, 2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['left_arm', [5]],
            4: ['right_arm', [6]],
            5: ['noise', [12]],
            6: ['neck', [13]]
        }
        parse_agnostic = parse_agnostic.argmax(dim=1)[:, None]  # 6,1,256,129
        parse_agnostic_old = parse_agnostic.float()

        parse_old = torch.zeros(parse_agnostic.size(0), opt.label_nc, opt.height, opt.width, dtype=torch.float).cuda()
        parse_agnostic = parse_agnostic.long().cuda()
        parse_old.scatter_(1, parse_agnostic, 1.0)
        parse = torch.zeros(parse_agnostic.size(0), 7, opt.height, opt.width, dtype=torch.float).cuda()


        parse_all = parse_all.argmax(dim=1)[:, None]
        parse_old_1 = torch.zeros(parse_all.size(0), opt.label_nc, opt.height, opt.width, dtype=torch.float).cuda()
        parse_all = parse_all.long().cuda()
        parse_old_1.scatter_(1, parse_all, 1.0)
        parse_a = torch.zeros(parse_all.size(0), 7, opt.height, opt.width, dtype=torch.float).cuda()

        for j in range(len(labels8)):
            for label in labels8[j][1]:
                parse[:, j] += parse_old[:, label]
                parse_a[:, j] += parse_old_1[:, label]


        pose_input = torch.cat([img_agnostic,parse, pose], dim=1)

        parse_pred = self.seg(c_masked,pose_input) #6,14,256,129
        parse_pred = torch.nn.functional.normalize(parse_pred)

        parse_pred1 = parse_pred.argmax(dim=1)[:, None]
        parse_pred_edge = torch.FloatTensor((parse_pred1.cpu().numpy() == 2).astype(np.int)).cuda()


        parse_a1 = parse_a.argmax(dim=1)[:, None]
        person_clothes_edge = torch.FloatTensor((parse_a1.cpu().numpy() == 2).astype(np.int))
        person_clothes = image * person_clothes_edge.cuda()
        person_clothes = person_clothes.cuda()
        person_clothes_edge = person_clothes_edge.cuda()

        # Part2:warp
        if opt.phase=='train':
            ref_input = torch.cat([person_clothes_edge, person_clothes_edge, person_clothes_edge], 1)
            gen_parse = parse_a
        else:
            ref_input = torch.cat([parse_pred_edge, parse_pred_edge, parse_pred_edge], 1)
            gen_parse = parse_pred

        tryon,warped_out, kp_source, kp_ref,dense_motion = self.warp(c_masked,person_clothes, pose, img_agnostic,ref_input,gen_parse,dropout_flag, dropout_p)

        return parse_pred,parse_pred_edge, parse_a,tryon,warped_out,person_clothes, ref_input, kp_source, kp_ref,dense_motion



class ResNet(nn.Module):
    def __init__(self,in_channels,num_classes=7):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        Layers = [3, 4]
        self.conv2 = self._make_layers(64,(64,64,256),Layers[0])
        self.conv3 = self._make_layers(256,(128,128,512),Layers[1],2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(512,10),
        )
        self.main_classifier = nn.Conv2d(512, num_classes, kernel_size=1)

        self.signoid = nn.Sigmoid()
    def forward(self,input):
        input_size = input.size()[2:]
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.main_classifier(x)
        x = F.upsample(x, input_size, mode='bilinear')  # upsample to the size of input image, scale=8
        return x

    def _make_layers(self,in_channels,filters,blocks,stride = 1):
        layers = []
        block_1 = Block(in_channels,filters,stride,is_1x1conv = True)
        layers.append(block_1)
        for i in range(1,blocks):
            print(filters[2])
            layers.append(Block(filters[2],filters,stride=stride,is_1x1conv=False))
        return nn.Sequential(*layers)

class Block(nn.Module):
    def __init__(self,in_channels,filters,stride,is_1x1conv = False):
        super(Block, self).__init__()
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(True)
        filter1,filter2,filter3 = filters
        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels,filter1,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2,kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU(True),
        )
        self.conv3= nn.Sequential(
            nn.Conv2d(filter2,filter3,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(filter3),

        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,filter3,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(filter3),
            )
    def forward(self,x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x =x + x_shortcut
        x = self.relu(x)
        return x
