import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from .util import TPS
from util import lpips
from PIL import Image, ImageDraw
from torchvision import transforms
import cv2

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


def simam(x, e_lambda=1e-4):
    b, c, h, w = x.shape
    n = w * h - 1
    x_minus_mu_square = (x - x.mean(axis=[2, 3], keepdim=True)) ** 2
    y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(axis=[2, 3], keepdim=True) / n + e_lambda)) + 0.5
    return x * nn.functional.sigmoid(y)

def show_keypoints(temp,img,keypoints,num_kp,num_tps,step):
    r = 2
    cv_img = (img.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    rgb = (cv_img * 255).astype(np.uint8)
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    unloader = transforms.ToPILImage()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = rgb
    image = unloader(image)

    for i in range(num_tps):
        for j in range(num_kp):
            number = i*num_kp + j
            pointx, pointy = keypoints[number]
            pointx = (pointx+1)*256-240
            pointy = (pointy+1)*192
            # pointx = pointx  * 192
            # pointy = pointy*256
            agnostic_draw = ImageDraw.Draw(image)
            agnostic_draw.rectangle((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'red', 'red')
            # img1 = transform(image)

            if temp == "soruce":
                # cv2.imwrite('./results/' + opt.name + '/' + str(step) + '.png', bgr)
                # cv2.imwrite('./temp/agnostic_' + str(step) + '_' + str(i)+'_source.png',bgr)
                image.save('temp/agnostic_' + str(step) + '_' + str(i) +'_source.png')
            else:
                # cv2.imwrite('./temp/agnostic_' + str(step) + '_' + str(i) + '_driving.png',bgr)
                image.save('temp/agnostic_' + str(step) + '_' + str(i) + '_driving.png')
        image = rgb
        image = unloader(image)
        # if i in [18,17,16,15]:
        #     agnostic_draw.rectangle((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'green', 'green')
        # elif i == 19:
        #     agnostic_draw.rectangle((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'red', 'red')
        # else:
        #     agnostic_draw.rectangle((pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2), 'gray', 'gray')
        # image.save('temp/agnostic_'+str(step)+'_source.png')
    # image.save('agnostic.png')

# def show_keypoints_driving(keypoints,img,step):
#     r = 2
#     unloader = transforms.ToPILImage()
#     image = img.cpu().clone()
#     image = unloader(image)
#     agnostic_draw = ImageDraw.Draw(image)
#     for i in range(len(keypoints)):
#         pointx, pointy = keypoints[i]
#         pointx = (pointx+1)*98
#         pointy = (pointy+1)*128
#         agnostic_draw.rectangle((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'green', 'green')
#         # if i in [18,17,16,15]:
#         #     agnostic_draw.rectangle((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'green', 'green')
#         # elif i == 19:
#         #     agnostic_draw.rectangle((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'red', 'red')
#         # else:
#         #     agnostic_draw.rectangle((pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2), 'gray', 'gray')
#         image.save('temp/agnostic_'+str(step)+'_driving.png')
# class MultiAffineRegularizationLoss(nn.Module):
#     def __init__(self, kz_dic):
#         super(MultiAffineRegularizationLoss, self).__init__()
#         self.kz_dic = kz_dic
#         self.method_dic = {}
#         for key in kz_dic:
#             instance = AffineRegularizationLoss(kz_dic[key])
#             self.method_dic[key] = instance
#         self.layers = sorted(kz_dic, reverse=True)
#
#     def __call__(self, flow_fields):
#         loss = 0
#         for i in range(len(flow_fields)):
#             method = self.method_dic[self.layers[i]]
#             loss += method(flow_fields[i])
#         return loss


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1.5, size_average=True, ignore_index=250):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, input, target):
        # F.cross_entropy(x,y)工作过程就是(Log_Softmax+NllLoss)：①对x做softmax,使其满足归一化要求，结果记为x_soft;②对x_soft做对数运算
        # 并取相反数，记为x_soft_log;③对y进行one-hot编码，编码后与x_soft_log进行点乘，只有元素为1的位置有值而且乘的是1，
        # 所以点乘后结果还是x_soft_log
        # 总之，F.cross_entropy(x,y)对应的数学公式就是CE(pt)=-1*log(pt)
        n, c, w, h = input.size()
        nt, wt, ht = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(
                ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        ce_loss = F.cross_entropy(
            input, target, weight=None, size_average=self.size_average, ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)  # pt是预测该类别的概率，要明白F.cross_entropy工作过程就能够理解
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.6, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        # pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, opt):
        noise = torch.normal(mean=0, std=opt.sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs
        self.tps = True
        self.control_points = make_coordinate_grid((opt.points_tps, opt.points_tps), type=noise.type())
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean=0,std=opt.sigma_tps * torch.ones([bs, 1, opt.points_tps ** 2]))


    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        # N * (HW) * 2 * 1
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            # 1 * (HW) * 1 * 2 - 1 * 1 * 25 *2
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out

class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def warploss(opt,source,warped_out,criterion_style,criterion_percept,criterionL1):
    loss_vgg = 0
    loss_values = {}
    loss_recnstruction = 0
    x = warped_out
    temp = 0
    for scale in range(4):
        temp += 1
        cur_source = F.interpolate(source, scale_factor=0.5 ** (3 - scale), mode='bilinear')
        loss_perceptual = criterion_percept(x[scale], cur_source).mean()
        loss_content, loss_style = criterion_style(x[scale], cur_source)
        loss_l1 = criterionL1(x[scale], cur_source)
        loss_vgg += temp*(loss_perceptual* opt.perceptual_weight + loss_style* opt.style_weight + loss_content* opt.content_weight)
        loss_recnstruction += loss_l1*temp*opt.l1_weight

    loss_values['loss_vgg'] = loss_vgg
    loss_values['l1'] = loss_recnstruction


    return loss_values





def kploss(opt,kp_extractor,source,ref,kp_source,kp_ref):
    transform_random = TPS(opt,mode='random', bs=ref.shape[0])
    ref_grid = transform_random.transform_frame(ref)
    transformed_ref = F.grid_sample(ref, ref_grid, padding_mode="reflection", align_corners=True)
    source_grid = transform_random.transform_frame(source)
    transformed_source = F.grid_sample(source, source_grid, padding_mode="reflection", align_corners=True)


    transformed_source ,transformed_ref = kp_extractor(transformed_source,transformed_ref)
    # generated = []
    # generated['transformed_frame'] = transformed_frame
    # generated['transformed_kp'] = transformed_kp

    warped = transform_random.warp_coordinates(transformed_ref)
    kp_d = kp_ref
    value = torch.abs(kp_d - warped).mean()
    loss_values = {}
    loss_values['equivariance_value_ref'] = opt.equivariance_value * value


    warped = transform_random.warp_coordinates(transformed_source)
    kp_d = kp_source
    value = torch.abs(kp_d - warped).mean()
    loss_values['equivariance_value_source'] = opt.equivariance_value * value
    return loss_values


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    # xx [[-1,-1/3,1/3,1],[-1,-1/3,1/3,1],[-1,-1/3,1/3,1],[-1,-1/3,1/3,1]]
    # yy [[-1,-1,-1,-1],[-1/3,-1/3,-1/3,-1/3],[1/3,1/3,1/3,1/3],[1,1,1,1]]

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

# class MultiAffineRegularizationLoss(nn.Module):
#     def __init__(self, kz_dic):
#         super(MultiAffineRegularizationLoss, self).__init__()
#         self.kz_dic=kz_dic
#         self.method_dic={}
#         for key in kz_dic:
#             instance = AffineRegularizationLoss(kz_dic[key])
#             self.method_dic[key] = instance
#         self.layers = sorted(kz_dic, reverse=True)
#
#     def __call__(self, flow_fields):
#         loss=0
#         for i in range(len(flow_fields)):
#             method = self.method_dic[self.layers[i]]
#             loss += method(flow_fields[i])
#         return loss



# class AffineRegularizationLoss(nn.Module):
#     """docstring for AffineRegularizationLoss"""
#     # kernel_size: kz
#     def __init__(self, kz):
#         super(AffineRegularizationLoss, self).__init__()
#         self.kz = kz
#         self.criterion = torch.nn.L1Loss()
#         self.extractor = BlockExtractor(kernel_size=kz)
#         self.reshape = LocalAttnReshape()
#
#         temp = np.arange(kz)
#         A = np.ones([kz*kz, 3])
#         A[:, 0] = temp.repeat(kz)
#         A[:, 1] = temp.repeat(kz).reshape((kz,kz)).transpose().reshape(kz**2)
#         AH = A.transpose()
#         k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz**2) #K = (A((AH A)^-1)AH - I)
#         self.kernel = np.dot(k.transpose(), k)
#         self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz**2, kz, kz).unsqueeze(1)
#
#     def __call__(self, flow_fields):
#         grid = self.flow2grid(flow_fields)
#
#         grid_x = grid[:,0,:,:].unsqueeze(1)
#         grid_y = grid[:,1,:,:].unsqueeze(1)
#         weights = self.kernel.type_as(flow_fields)
#         loss_x = self.calculate_loss(grid_x, weights)
#         loss_y = self.calculate_loss(grid_y, weights)
#         return loss_x+loss_y
#
#
#     def calculate_loss(self, grid, weights):
#         results = nn.functional.conv2d(grid, weights)   # KH K B [b, kz*kz, w, h]
#         b, c, h, w = results.size()
#         kernels_new = self.reshape(results, self.kz)
#         f = torch.zeros(b, 2, h, w).type_as(kernels_new) + float(int(self.kz/2))
#         grid_H = self.extractor(grid, f)
#         result = torch.nn.functional.avg_pool2d(grid_H*kernels_new, self.kz, self.kz)
#         loss = torch.mean(result)*self.kz**2
#         return loss
#
#     def flow2grid(self, flow_field):
#         b,c,h,w = flow_field.size()
#         x = torch.arange(w).view(1, -1).expand(h, -1).type_as(flow_field).float()
#         y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(flow_field).float()
#         grid = torch.stack([x,y], dim=0)
#         grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
#         return flow_field+grid




class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, for_dis=None):
        if self.type == 'hinge':
            if for_dis:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return content_loss, style_loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


# class PerceptualCorrectness(nn.Module):
#     r"""
#
#     """
#
#     def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1']):
#         super(PerceptualCorrectness, self).__init__()
#         self.add_module('vgg', VGG19())
#         self.layer = layer
#         self.eps=1e-8
#         self.resample = Resample2d(4, 1, sigma=2)
#
#     def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=False):
#         used_layers=sorted(used_layers, reverse=True)
#         # self.target=target
#         # self.source=source
#         self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
#         loss = 0
#         for i in range(len(flow_list)):
#             loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)
#
#
#
#         return loss
#
#     def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
#         target_vgg = self.target_vgg[layer]
#         source_vgg = self.source_vgg[layer]
#         [b, c, h, w] = target_vgg.shape
#
#         # maps = F.interpolate(maps, [h,w]).view(b,-1)
#         flow = F.interpolate(flow, [h,w])
#
#         target_all = target_vgg.view(b, c, -1)                      #[b C N2]
#         source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]
#
#
#         source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
#         target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
#         try:
#             correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
#         except:
#             print("An exception occurred")
#             print(source_norm.shape)
#             print(target_norm.shape)
#         (correction_max,max_indices) = torch.max(correction, dim=1)
#
#         # interple with bilinear sampling
#         if use_bilinear_sampling:
#             input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
#         else:
#             input_sample = self.resample(source_vgg, flow).view(b, c, -1)
#
#         correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
#         loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
#         if mask is None:
#             loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
#         else:
#             mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
#             mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3))
#             loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
#             loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)
#
#         # print(correction_sample[0,2076:2082])
#         # print(correction_max[0,2076:2082])
#         # coor_x = [32,32]
#         # coor = max_indices[0,32+32*64]
#         # coor_y = [int(coor%64), int(coor/64)]
#         # source = F.interpolate(self.source, [64,64])
#         # target = F.interpolate(self.target, [64,64])
#         # source_i = source[0]
#         # target_i = target[0]
#
#         # source_i = source_i.view(3, -1)
#         # source_i[:,coor]=-1
#         # source_i[0,coor]=1
#         # source_i = source_i.view(3,64,64)
#         # target_i[:,32,32]=-1
#         # target_i[0,32,32]=1
#         # lists = str(int(torch.rand(1)*100))
#         # img_numpy = util.tensor2im(source_i.data)
#         # util.save_image(img_numpy, 'source'+lists+'.png')
#         # img_numpy = util.tensor2im(target_i.data)
#         # util.save_image(img_numpy, 'target'+lists+'.png')
#         return loss
#
#     def bilinear_warp(self, source, flow):
#         [b, c, h, w] = source.shape
#         x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
#         y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
#         grid = torch.stack([x,y], dim=0)
#         grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
#         grid = 2*grid - 1
#         flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
#         grid = (grid+flow).permute(0, 2, 3, 1)
#         input_sample = F.grid_sample(source, grid).view(b, c, -1)
#         return input_sample



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
