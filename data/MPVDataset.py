import os
import json

import cv2
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
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
# from draw_keypoints import get_poseimg

def CreateDataset(opt):
    # from data.aligned_dataset import AlignedDataset
    dataset = MPVDataset()
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

class MPVDataset(Dataset):
    def initialize(self, opt):
        super(MPVDataset, self).__init__()

        self.opt = opt
        self.data_path = os.path.join(opt.dataroot,'MPV')
        self.diction = {}
        self.img_size = [256,192]
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
        self.semantic_nc = opt.label_nc
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_pairs = None
        
        self.opt = opt
        self.phase = opt.phase

        self.filepath_df = pd.read_csv(os.path.join(self.data_path, "all_poseA_poseB_clothes.txt"), sep="\t", names=["poseA", "poseB", "target", "split"])
        self.filepath_df = self.filepath_df.drop_duplicates("poseA")
        self.filepath_df = self.filepath_df[self.filepath_df["poseA"].str.contains("front")]
        # self.filepath_df = self.filepath_df.drop(["poseB"], axis=1)
        self.filepath_df = self.filepath_df.sort_values("poseA")
        self.warp_clean = opt.warp_clean
        opt.train_size = 0.9
        opt.val_size = 0.1
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        if self.phase == "test_same":
            self.filepath_df = self.filepath_df[self.filepath_df.split == "test"]
            
        if self.phase == "test":
            if opt.warp_clean == True:
                self.filepath_df = pd.read_csv(os.path.join(self.data_path, "test_CTP.txt"), sep=" ",
                                               names=["poseA", "target"])
                self.filepath_df = self.filepath_df.sort_values("poseA")
            else:

                self.filepath_df = pd.read_csv(os.path.join(self.data_path, "test_PTP.txt"), sep="  ",
                                              names=["poseA", "poseB"])
        elif self.phase == "train":
            self.filepath_df = self.filepath_df[self.filepath_df.split == "train"]


    def get_parse_agnostic(self, parse):
        parse_array = np.array(parse)

        agnostic = parse.copy()
        parse_need = ((parse_array == 0).astype(np.float32) +
                      (parse_array == 5).astype(np.float32) +
                      (parse_array == 7).astype(np.float32) +
                      (parse_array == 20).astype(np.float32) +
                      (parse_array == 14).astype(np.float32) +
                      (parse_array == 15).astype(np.float32)
                      )
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_need * 255), 'L'))
        return agnostic


    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]

        # get original image of person
        image = Image.open(osp.join(self.data_path, 'all', df_row["poseA"])).convert('RGB')
        image = transforms.Resize(self.fine_width, interpolation=2)(image)
        image = self.img_transform(image)

        # img_agnostic
        img_agnostic = Image.open(osp.join(self.data_path,'img_agnostic', df_row["poseA"])).convert('RGB')
        img_agnostic = transforms.Resize(self.fine_width, interpolation=2)(img_agnostic)
        img_agnostic = self.img_transform(img_agnostic)


        # extract non-warped cloth
        if self.warp_clean ==True:
            cloth_image = Image.open(os.path.join(self.data_path,'all', df_row["target"])).convert('RGB')
            c_mask = Image.open(os.path.join(self.data_path, 'all', df_row["target"][:-4] + "_mask.jpg"))
            c_array = np.array(cloth_image)
            target = self.img_transform(cloth_image)
        else:
            person_B = Image.open(os.path.join(self.data_path, 'all',df_row["poseB"])).convert('RGB')
            target = transforms.Resize(self.fine_width, interpolation=2)(person_B)
            target = self.img_transform(target)

            parse_B = Image.open(osp.join(self.data_path, 'parse', df_row["poseB"][:-4] + "_parse.png"))
            parse_carray = np.array(parse_B)
            parse_need = torch.FloatTensor(
                (parse_carray == 5).astype(np.int) +
                (parse_carray == 7).astype(np.int))
            blank = Image.fromarray(np.ones((256, 192, 3), np.uint8) * 128)
            mask = np.repeat(np.expand_dims(parse_need, -1), 3, axis=-1).astype(np.uint8)
            c_array = blank * (1 - mask) + mask * person_B
            cloth_image = Image.fromarray(np.uint8(c_array))

        # load  labels
        parse_A = Image.open(os.path.join(self.data_path, 'parse', df_row["poseA"][:-4] + "_parse.png"))
        parse_array = np.array(parse_A)
        parse_roi = (parse_array == 5).astype(np.float32) + (parse_array == 7).astype(np.float32)
        # (parse_array == 14).astype(np.float32) + \
        # (parse_array == 15).astype(np.float32)
        parse = parse_A.convert('P')
        parse_all = torch.from_numpy(np.asarray(parse_A)[None]).long()

        parse_agnostic = self.get_parse_agnostic(parse)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        # CATEGORY = (
        #     0:'Background', 1'Hat', 2.Hair', 3'Glove', 4'Sunglasses',
        #     5'Upper-clothes' ,6'Pants', 7'Coat',8'Socks' , 9'Left-shoe',
        #     10'Right-shoe', 11'Scarf', ,12'Dress',  13'Face' , 14'Left-arm',
        #     15'Right-arm',16'Left-leg', 17'Right-leg', 18'tosor-skin',19'Skirt',20'neck'
        # )
        labels = {
            0: ['background', [0]],
            1: ['hair', [1, 2, 3]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 7]],
            4: ['bottom', [ 12, 6]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [9]],
            10: ['right_shoe', [10]],
            11: ['socks', [8]],
            12: ['noise', [11,19,18]],
            13: ['neck', [20]]
        }
        parse_agnostic_map = torch.zeros(21, self.fine_height, self.fine_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(14, self.fine_height, self.fine_width, dtype=torch.float)

        parse_map = torch.zeros(21, self.fine_height, self.fine_width, dtype=torch.float)
        parse_map.scatter_(0, parse_all, 1.0)
        new_parse_all_map = torch.zeros(14, self.fine_height, self.fine_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_all_map[i] += parse_map[label]
                new_parse_agnostic_map[i] += parse_agnostic_map[label]


        align_factor = 1.0
        if self.warp_clean == True:
            c_fg = np.where(c_array != 255)
        else:
            c_fg = np.where(c_array != 128)
        t_c, b_c = min(c_fg[0]), max(c_fg[0])
        l_c, r_c = min(c_fg[1]), max(c_fg[1])
        c_bbox_center = [(l_c + r_c) / 2, (t_c + b_c) / 2]
        c_bbox_w = r_c - l_c  # 衣服宽
        c_bbox_h = b_c - t_c

        parse_roi_fg = np.where(parse_roi != 0)
        parse_roi_fg = list(parse_roi_fg)

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

        clothes = cloth_image.resize((int(cloth_image.size[0] * scale_factor), int(cloth_image.size[1] * scale_factor)),
                               Image.BILINEAR)
        black = Image.fromarray(np.ones((256, 192, 3), np.uint8) * 128)
        black.paste(clothes, (paste_x, paste_y))
        cloth_image = black

        if self.warp_clean == True:
            white_c = Image.fromarray(np.ones((256, 192, 3), np.uint8) * 255)
            white_c.paste(clothes, (paste_x, paste_y))
            cloth_image = white_c
            c_mask = c_mask.resize((int(c_mask.size[0] * scale_factor), int(c_mask.size[1] * scale_factor)),
                                   Image.BILINEAR)
            black_c = Image.fromarray(np.ones((256, 192, 3), np.uint8) * 0)
            black_c.paste(c_mask, (paste_x, paste_y))


        masked_cloth = self.img_transform(cloth_image)
        if self.warp_clean == True:
            black_c = np.array(black_c)
            black_c = self.transform(black_c)
            masked_cloth = masked_cloth * black_c

        # load pose points
        pose_name = df_row["poseA"].replace('.jpg', '_keypoints.jpg')
        pose_rgb = Image.open(osp.join(self.data_path,'keypoints_map', pose_name))
        pose_rgb = transforms.Resize(self.fine_width, interpolation=2)(pose_rgb)
        pose_rgb = self.img_transform(pose_rgb)


        return {"image": image,
                "cloth": masked_cloth,
                "img_agnostic": img_agnostic,
                'pose':pose_rgb,
                "parse": new_parse_all_map,
                "parse_agnostic": new_parse_agnostic_map,
                "img_name": df_row["poseA"],
                "target":target,

        }
            
    
    def __len__(self):
        return len(self.filepath_df)
    
    def name(self):
        return "MPVDataset"

class MPVDataLoader:
    def __init__(self, opt, dataset):
        super(MPVDataLoader, self).__init__()

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