import torch
import torch.nn as nn
from utils.network import KernelConv
import utils.utils as kpn_utils
import numpy as np
from src.utils import MemoryEfficientSwish
import torch.nn.functional as F
from utils.rdca import RDCA
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models



class FE(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rate_1, dilation_rate_2, dilation_rate_3, epsilon=1e-4):
        super(FE, self).__init__()
        self.epsilon = epsilon
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate_1, dilation=dilation_rate_1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate_2, dilation=dilation_rate_2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate_3, dilation=dilation_rate_3)
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.activation = MemoryEfficientSwish()

        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 4, 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        weight_map = self.weight_calc(x)
        x = weight_map[:, 0:1, :, :] * x1 + \
            weight_map[:, 1:2, :, :] * x2 + \
            weight_map[:, 2:3, :, :] * x3 + \
            weight_map[:, 3:4, :, :] * x4
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, epsilon=1e-4):
        super(ConvBlock, self).__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.activation = MemoryEfficientSwish()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x1, x2, upsample):
        weight = self.relu(self.weights)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        if upsample:
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x2 = F.interpolate(x2, scale_factor=1/2, mode='bilinear', align_corners=True)
        x = weight[0] * x1 + weight[1] * x2
        x = self.activation(x)
        x = self.conv(x)
        return x

class ASFF(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASFF, self).__init__()
        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels*3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, f1, f2, f3):
        """
        f3: 64*64
        f2: 128*128
        f1: 256*256
        """
        f1 = self.conv1(f1)
        f3 = F.interpolate(f3, scale_factor=2, mode='nearest')
        weights = torch.cat([f1, f2, f3], 1)
        weights = self.weight_calc(weights)
        f2 = weights[:, 0:1, :, :] * f1 + \
             weights[:, 1:2, :, :] * f2 + \
             weights[:, 2:3, :, :] * f3
        f2 = self.conv(f2)
        return f2


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class MSFF(nn.Module):
    '''Multi-scale feature aggregation.'''

    def __init__(self, epsilon=1e-4):
        super(MSFF, self).__init__()

        self.epsilon = epsilon
        self.fe1 = FE(64, 64, 4, 6, 8)
        self.fe2 = FE(128, 64, 2, 4, 6)
        self.fe3 = FE(256, 64, 1, 2, 4)

        self.conv1_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv1_2 = ConvBlock(64, 64)
        self.conv2_1 = ConvBlock(64, 64)
        self.conv2_2 = ConvBlock(64, 64)
        self.conv3_1 = ConvBlock(64, 64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.asff = ASFF(64, 128)



    def forward(self, f1, f2, f3):
        """
        f1  64*256*256
        f2  128*128*128
        f3  256*64*64
        """
        f1 = self.fe1(f1)
        f2 = self.fe2(f2)
        f3 = self.fe3(f3)

        f3 = self.conv1_1(f3)
        f2 = self.conv2_1(f2, f3, True)
        f1 = self.conv3_1(f1, f2, True)
        f1 = self.conv3_2(f1)
        f2 = self.conv2_2(f2, f1, False)
        f3 = self.conv1_2(f3, f2, False)

        f_2 = self.asff(f1, f2, f3)

        return f_2


class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True,
                 in_channels=128, out_channels=128, dilation_rate_list=[1, 2, 4, 8]):
        super(InpaintGenerator, self).__init__()

        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size

        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        self.msff = MSFF()
        self.rdca = RDCA()

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()

        if init_weights:
            self.init_weights()

    def forward(self, x, mask):


        inputs = x.clone()

        x = self.encoder0(x)  # 64*256*256

        x1 = x.clone()

        x = self.encoder1(x)  # 128*128*128
        x2 = x.clone()

        x = self.encoder2(x)  # 256*64*64
        x3 = x.clone()

        x2 = self.msff(x1, x2, x3)

        kernels_img, foreground = self.kpn_model(inputs, x2)

        output_f = self.rdca(x, foreground, mask)

        x = self.middle(output_f )  # 256*64*64

        x = self.decoder(x)  # 3*256*256

        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        x = (torch.tanh(x) + 1) / 2

        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
