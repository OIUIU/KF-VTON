import argparse
import os
from util import util
import torch
import models
import data

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='tryon', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        # self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        self.parser.add_argument("--no_bg", action='store_true', default=False,
                            help="whether to remove the background in I_m")
        self.parser.add_argument('--warp_clean', action='store_true', default=False)

        # input/output sizes
        self.parser.add_argument('--height', type=int, default=256)
        self.parser.add_argument('--width', type=int, default=192)
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--crop_size', type=int, default=256,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        self.parser.add_argument('--label_nc', type=int, default=14, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str,default='dataset')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')


        self.parser.add_argument('--init_type',
                                 choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'],
                                 default='xavier')
        self.parser.add_argument('--init_variance', type=float, default=0.02,
                                 help='variance of the initialization distribution')


        #kp detector
        self.parser.add_argument('--num_tps', type=int, default=8, help='number of keypoints')
        self.parser.add_argument('--num_kp', type=int, default=12, help='number of root keypoints')

        # for generator
        self.parser.add_argument('--gen_semantic_nc', type=int, default=7,help='# 7of input label classes without unknown class'
                                                                               '8 for pose transfer')

        self.parser.add_argument('--ngf', type=int, default=32, help='32# of gen filters in first conv layer')
        self.parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='normal',
                            help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                                 'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
        self.parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')


        # discriminator
        self.parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')



        #pose transfer
        self.parser.add_argument('--train_dir', type=str, default='./sample')
        self.parser.add_argument('--dataset_mode', type=str, default='fashion')
        # self.parser.add_argument('--PoseTransfer', type=str, default='pose')
        self.parser.add_argument('--posedataroot', type=str, default='hd/fashion_data')
        self.parser.add_argument('--load_size', type=int, default=256,
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        self.parser.add_argument('--posemodel', type=str, default='posetransfer')
        # display parameter define
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='display id of the web')
        self.parser.add_argument('--display_port', type=int, default=8096, help='visidom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                            help='if positive, display all images in a single visidom web panel')
        self.parser.add_argument('--display_env', type=str, default=self.parser.parse_known_args()[0].name.replace('_', ''),
                            help='the environment of visidom display')
        self.parser.add_argument('--angle', type=float, default=False)
        self.parser.add_argument('--shift', type=float, default=False)
        self.parser.add_argument('--scale', type=float, default=False)
        self.parser.add_argument('--old_size', type=int, default=(256, 256),
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        self.parser.add_argument('--structure_nc', type=int, default=18)
        self.initialized = True


    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        self.opt.semantic_nc = 1
        if torch.cuda.is_available():
            self.opt.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            self.opt.device = torch.device("cpu")


        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        # if save and not self.opt.continue_train:
        #     file_name = os.path.join(expr_dir, 'opt.txt')
        #     with open(file_name, 'wt') as opt_file:
        #         opt_file.write('------------ Options -------------\n')
        #         for k, v in sorted(args.items()):
        #             opt_file.write('%s: %s\n' % (str(k), str(v)))
        #         opt_file.write('-------------- End ----------------\n')
        return self.opt
