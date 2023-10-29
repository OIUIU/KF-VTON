from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--RESUME',default=False,action='store_true')
        self.parser.add_argument('--local_rank', type=int, default=0)

        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        self.parser.add_argument('--save_iters_freq', type=int, default=20000, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_iterslatest_freq', type=int, default=20000,
                            help='frequency of saving the latest results')
        self.parser.add_argument('--eval_iters_freq', type=int, default=15000,
                            help='frequency of showing training results on screen')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--lr_seg', type=float, default=0.0001, help='initial learning rate for adam')
        # self.parser.add_argument('--lr_kp_detector', type=float, default=0.000005, help='initial learning rate for adam')
        self.parser.add_argument('--lr_warp', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for adam')
        # for discriminators
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        self.parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')

        #anchor
        self.parser.add_argument('--sigma_affine', type=int, default=0.05, help='Sigma for affine part')
        self.parser.add_argument('--sigma_tps', type=int, default=0.005, help='Sigma for deformation part')
        self.parser.add_argument('--points_tps', type=int, default=6, help='nNumber of point in the deformation grid')
        self.parser.add_argument('--equivariance_value', type=int, default=100, help='Weights for value equivariance.')

        #dropout
        self.parser.add_argument('--dropout_epoch', type=int, default=35, help='The first dropout_epoch training uses dropout operation')
        self.parser.add_argument('--dropout_inc_epoch', type=int, default=10,help='The probability P will linearly increase from dropout_startp to dropout_maxp in dropout_inc_epoch epochs')
        self.parser.add_argument('--dropout_maxp', type=int, default=0.7)
        self.parser.add_argument('--dropout_startp', type=int, default=0.0)

        #warp
        self.parser.add_argument('--perceptual_weight', type=int, default=10,help='Weights for perceptual loss')
        self.parser.add_argument('--l1_weight', type=int, default=10,help='Weights for reconstruction loss')
        self.parser.add_argument('--content_weight', type=int, default=1, help='Weights for content loss')
        self.parser.add_argument('--style_weight', type=int, default=100, help='Weights for style loss')

        self.isTrain = True
        self.parser.add_argument('--dataset_list', type=str, default='train_pairs.txt')
        self.parser.add_argument("--load_step", type=int, default=0)
        self.parser.add_argument("--keep_step", type=int, default=100000)
        self.parser.add_argument("--decay_step", type=int, default=100000)

