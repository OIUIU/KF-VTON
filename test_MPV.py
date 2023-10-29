import time
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
import cv2
import datetime
from data.MPVDataset import MPVDataset, MPVDataLoader,CreateDataset
from util.util import get_palette
from PIL import Image
from models.network import load_checkpoint
from models.keypoint_detector import KPDetector
from options.test_options import TestOptions
from models.tryon import TryOnNetwork
from models.KPWarp import WarpNetwork
from models.seg_network import SegGenerator
from models.external_function import show_keypoints
from util import flow_util
from util.flow_util import de_offset
def main():
    opt = TestOptions().parse()
    os.makedirs('sample',exist_ok=True)

    start_epoch, epoch_iter = 1, 0
    f2c = flow_util.flow2color()
    test_data = CreateDataset(opt)
    test_loader = MPVDataLoader(opt, test_data)
    test_dataset = test_loader.data_loader
    dataset_size = len(test_data) / opt.batchSize
    print('#testing images = %d' % dataset_size)
    seg_model = SegGenerator(opt, input_A=13, input_B=3,output_nc=7)
    warp_model = WarpNetwork(opt, block_expansion=64, max_features=256,gen_input=9)
    generator_full = TryOnNetwork(opt, seg_model, warp_model)

    generator_full.cuda()
    generator_full.eval()
    load_checkpoint(seg_model,opt.seg_checkpoint)
    load_checkpoint(warp_model, opt.warp_checkpoint)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0

    for epoch in range(1,2):
        for i, data in enumerate(test_dataset, start=int(epoch_iter)):
            total_steps += 1
            epoch_iter += 1

            img_names = data['img_name']
            image = data['image'].cuda()
            img_agnostic = data['img_agnostic'].cuda()
            parse_agnostic = data['parse_agnostic'].cuda()
            parse_all = data['parse'].cuda()
            pose = data['pose'].cuda()
            clothes = data['cloth'].cuda()
            target = data['target'].cuda()

            dropout_flag = False
            dropout_p = 0
            with torch.no_grad():
                parse_pred, parse_pred_edge, parse_a, tryon, warped_clothes, person_clothes, cond_input, kp_source, kp_ref, dense_motion = generator_full(
                    opt, image, img_agnostic, parse_agnostic,
                    parse_all, pose, clothes, dropout_flag, dropout_p, step)


            _, _, h, w = tryon.shape
            deformation = dense_motion['deformation']
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h,w), mode='bilinear', align_corners=True)
            category = ('background', 'paste', 'upper', 'left_arm', 'right_arm', 'noise','neck')
            palette = get_palette(len(category))
            os.makedirs('results', exist_ok=True)
            path = 'results/' + opt.name
            os.makedirs(path, exist_ok=True)

            if step % 1 == 0:
                a = image.cuda()
                f = target.cuda()
                b = clothes.cuda()
                c = warped_clothes.cuda()
                d = tryon.cuda()
                e = img_agnostic
                combine = torch.cat([a[0],f[0],d[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('results/' + opt.name + '/' + img_names[0].split('/')[-1], bgr)


            step += 1
            if epoch_iter >= dataset_size:
                break

if __name__ == '__main__':
    main()


