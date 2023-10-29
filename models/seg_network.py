import torch
import torch.nn as nn
from models.network import BaseNetwork
from options.train_options import TrainOptions
from models.util import ResBlock2d
from torch.nn import functional as F
opt = TrainOptions().parse()
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from functools import partial


#

class SegGenerator(BaseNetwork):
    def __init__(self, opt, input_A, input_B,output_nc=7,ngf = 64, norm_layer=nn.InstanceNorm2d):
        super(SegGenerator, self).__init__()

        self.pose_Encoder = nn.Sequential(
            ResBlock(input_A, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )
        self.cloth_Encoder = nn.Sequential(
            ResBlock(input_B, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf * 1, ngf * 1, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf * 1, ngf * 1, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        self.Decoder = nn.Sequential(
            ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
            ResBlock(ngf * 12, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
            ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 64
            ResBlock(ngf * 8, ngf * 2, norm_layer=norm_layer, scale='up'),  # 128
            ResBlock(ngf * 4, ngf, norm_layer=norm_layer, scale='up')  # 256
        )

        self.out_layer = ResBlock(ngf + input_A + input_B, output_nc, norm_layer=norm_layer, scale='same')
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)
        self.phase = opt.phase
        if self.phase == 'train':
            self.old_lr = opt.lr_seg

        self.Coattention = CoattentionModel(all_channel=256)


    def forward(self, cloth, pose, upsample='bilinear'):
        pose_list = []
        cloth_list = []
        for i in range(5):
            if i == 0:
                pose_list.append(self.pose_Encoder[i](pose))
                cloth_list.append(self.cloth_Encoder[i](cloth))
            else:
                pose_list.append(self.pose_Encoder[i](pose_list[i - 1]))
                cloth_list.append(self.cloth_Encoder[i](cloth_list[i - 1]))

        for i in range(5):
            N, _, iH, iW = pose_list[4 - i].size()

            if i == 0:
                T1 = pose_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = cloth_list[4 - i]

                T1, T2 = self.Coattention(T1, T2)
                feature = self.Decoder[i](torch.cat([T1, T2], 1))

            else:
                if i < 3:
                    pose_list[4 - i], cloth_list[4 - i] = self.Coattention(pose_list[4 - i], cloth_list[4 - i])
                    T1 = self.conv1[4 - i](F.interpolate(T1, scale_factor=2, mode=upsample) + pose_list[4 - i])
                    T2 = self.conv2[4 - i](F.interpolate(T2, scale_factor=2, mode=upsample) + cloth_list[4 - i])

                else:
                    T1 = pose_list[4 - i]
                    T2 = cloth_list[4 - i]

                feature = self.Decoder[i](torch.cat([T1, T2, feature], 1))
        x = self.out_layer(torch.cat([feature, cloth, pose], 1))
        if self.phase == "train":
            x = self.sigmoid(x)

        return x

    def update_learning_rate(self, optimizer):
        lrd = opt.lr_seg / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class CoattentionModel(nn.Module):
    def __init__(self,all_channel, all_dim=60 * 60):  # 473./8=60
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.softmax = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2):
        exemplar = input1
        query = input2
        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat) #HW*HW
        A1 = F.softmax(A.clone(), dim=1)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        return input1_att, input2_att  # shape: NxCx


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)

        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))
