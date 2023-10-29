from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--warp_checkpoint', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--seg_checkpoint', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--dataset_list', type=str, default='')
        self.isTrain = False