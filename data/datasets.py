import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from data.base_dataset import kp_to_map, BaseDataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from util.util import gen_noise,get_palette
def CreateDataset(opt):
    # from data.aligned_dataset import AlignedDataset
    dataset = VITONDataset()
    print("hd [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    print (dir, f)
    dirs= os.listdir(dir)
    for img in dirs:

        path = os.path.join(dir, img)
        #print(path)
        images.append(path)
    return images

class VITONDataset(data.Dataset):
    def initialize(self, opt):
        super(VITONDataset, self).__init__()
        self.opt = opt
        self.data_path = os.path.join(opt.dataroot, 'VTON',opt.phase)
        self.diction={}
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
        self.semantic_nc = opt.label_nc
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        # load data list
        img_names = []
        c_names = []
        with open(osp.join(opt.dataroot,'VTON', opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names = dict()
        self.c_names['paired'] = c_names
        self.c_names['unpaired'] = c_names


    def get_parse_agnostic(self, parse):
        parse_array = np.array(parse)

        agnostic = parse.copy()

        parse_need = ((parse_array == 0).astype(np.float32) +
                (parse_array == 5).astype(np.float32) +
                (parse_array == 7).astype(np.float32) +
                (parse_array == 10).astype(np.float32) +
                (parse_array == 14).astype(np.float32) +
                (parse_array == 15).astype(np.float32)
        )
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_need * 255), 'L'))
        return agnostic

    def get_img_agnostic(self,img, parse):
        parse_array = np.array(parse)
        agnostic = parse.copy()

        mask = (
                # (parse_array == 0).astype(np.float32) +
                (parse_array == 5).astype(np.float32) +
                (parse_array == 7).astype(np.float32) +
                (parse_array == 14).astype(np.float32) +
                (parse_array == 15).astype(np.float32)+
                (parse_array == 10).astype(np.float32)
        )
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1).astype(np.uint8)
        img_agnostic = img * (1 - mask)
        # agnostic.paste(0, None, Image.fromarray(np.uint8(parse_need * 255), 'L'))


        return img_agnostic

    def __getitem__(self, index):

        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        pc = {}
        align_factor = 1.0

        pose_name = img_name.replace('.jpg', '_keypoints.jpg')
        pose_rgb = Image.open(osp.join(self.data_path, 'pose', pose_name))
        pose_rgb = transforms.Resize(self.fine_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)



        # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'label', parse_name))
        parse_array = np.array(parse)
        parse_roi =(parse_array == 5).astype(np.float32) + (parse_array == 7).astype(np.float32)
        # (parse_array == 14).astype(np.float32) + \
        # (parse_array == 15).astype(np.float32)
        parse = parse.convert('P')
        parse_all = torch.from_numpy(np.asarray(parse)[None]).long()


        parse_agnostic = self.get_parse_agnostic(parse)
        # parse_agnostic.save('agnostic.png')
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()


        # CATEGORY = (
        #     0:'Background', 1'Hat', 2.Hair', 3'Glove', 4'Sunglasses',
        #     5'Upper-clothes', 6'Dress', 7'Coat',8'Socks', 9'Pants',
        #     10'Jumpsuits', 11'Scarf', 12'Skirt', 13'Face', 14'Left-arm',
        #     15'Right-arm',16'Left-leg', 17'Right-leg', 18'Left-shoe', 19'Right-shoe'
        # )
        labels = {
            0: ['background', [0]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 7]],
            4: ['bottom', [9, 12 ,6]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]],
            13: ['neck',[10]]
        }
        parse_agnostic_map = torch.zeros(20, self.fine_height, self.fine_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(14, self.fine_height, self.fine_width, dtype=torch.float)


        parse_map = torch.zeros(20, self.fine_height, self.fine_width, dtype=torch.float)
        parse_map.scatter_(0, parse_all, 1.0)
        new_parse_all_map = torch.zeros(14, self.fine_height, self.fine_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_all_map[i] += parse_map[label]
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # load person image

        img = Image.open(osp.join(self.data_path, 'img', img_name))
        img = transforms.Resize(self.fine_width, interpolation=2)(img)
        img_agnostic = Image.open(osp.join(self.data_path, 'agnostic', img_name))
        img_agnostic = transforms.Resize(self.fine_width, interpolation=2)(img_agnostic)

        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]


        for key in self.c_names:
            if key == "unpaired":
                continue
            else:
                c_name[key] = self.c_names[key][index].replace("_0.jpg","_1.jpg")
            if self.opt.warp_clean == False:
                c_name[key] = self.c_names[key][index].replace("_1.jpg", "_0.jpg")
                c[key] = Image.open(osp.join(self.data_path, 'img', c_name[key])).convert('RGB')
                pc[key] = c[key]
                pc[key] = transforms.Resize(self.fine_width, interpolation=2)(pc[key])
                pc[key] = self.transform(pc[key])

                parse_cname = c_name[key].replace('.jpg', '.png')
                parse_c = Image.open(osp.join(self.data_path, 'label', parse_cname))
                parse_carray = np.array(parse_c)
                parse_need = torch.FloatTensor(
                    (parse_carray == 5).astype(np.int) +
                    (parse_carray == 7).astype(np.int))
                blank = Image.fromarray(np.ones((256, 192, 3), np.uint8) * 128)
                mask = np.repeat(np.expand_dims(parse_need, -1), 3, axis=-1).astype(np.uint8)
                c_array = blank * (1 - mask) +mask*c[key]

                c[key] = Image.fromarray(np.uint8(c_array))
            else:
                c[key] = Image.open(osp.join(self.data_path, 'clothes', c_name[key])).convert('RGB')
                c_array = np.array(c[key])

            c_fg = np.where(c_array != 128)#128
            t_c, b_c = min(c_fg[0]), max(c_fg[0])
            l_c, r_c = min(c_fg[1]), max(c_fg[1])
            c_bbox_center = [(l_c + r_c) / 2, (t_c + b_c) / 2]
            c_bbox_w = r_c - l_c  # 衣服宽
            c_bbox_h = b_c - t_c

            parse_roi_fg = np.where(parse_roi != 0)
            t_parse_roi, b_parse_roi = min(parse_roi_fg[0]), max(parse_roi_fg[0])
            l_parse_roi, r_parse_roi = min(parse_roi_fg[1]), max(parse_roi_fg[1])
            parse_roi_bbox_w = r_parse_roi - l_parse_roi
            parse_roi_bbox_h = b_parse_roi - t_parse_roi
            parse_roi_center = [(l_parse_roi + r_parse_roi) / 2, (t_parse_roi + b_parse_roi) / 2]

            if c_bbox_w / c_bbox_h > parse_roi_bbox_w / parse_roi_bbox_h:
                ratio = parse_roi_bbox_h / c_bbox_h
                scale_factor = ratio * align_factor
            else:
                ratio = parse_roi_bbox_w / c_bbox_w
                scale_factor = ratio * align_factor
            paste_x = int(parse_roi_center[0] - c_bbox_center[0] * scale_factor)
            paste_y = int(parse_roi_center[1] - c_bbox_center[1] * scale_factor)

            c[key] = c[key].resize((int(c[key].size[0] * scale_factor), int(c[key].size[1] * scale_factor)),
                                   Image.BILINEAR)
            blank_c = Image.fromarray(np.ones((256, 192, 3), np.uint8) * 128)
            blank_c.paste(c[key], (paste_x, paste_y))
            c[key] = blank_c

            c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])
            c[key] = self.transform(c[key])



        result = {
            'img_name': img_name,
            'c_name': c_name,
            'image': img,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'parse': new_parse_all_map,
            'pose': pose_rgb,
            'cloth': c,
            'target': pc,

        }
        return result



    def __len__(self):
        # return len(self.A_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)
        return len(self.img_names)

    def name(self):
        return 'VITONDataset'

class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        # if opt.shuffle:
        #     train_sampler = data.sampler.RandomSampler(hd)
        # else:
        #     train_sampler = None

        self.data_loader = data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False,
                num_workers=1, pin_memory=True, drop_last=True)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
    # def __len__(self):
    #     return len(self.batch_sampler)