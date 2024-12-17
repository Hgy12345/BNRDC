import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.odconv import ODConv2d
from src.utils import MemoryEfficientSwish

import numpy as np


def extract_patches(x, kernel_size=3, stride=1):
    if kernel_size != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()

def odconv(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=4, padding=1, kernel_size=3):
    return ODConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                    reduction=reduction, kernel_num=kernel_num)

class RAL(nn.Module):
    '''Region affinity learning.'''

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super(RAL, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale

        self.conv_c = odconv(in_planes=256, out_planes=256)
        self.conv_m = odconv(in_planes=256, out_planes=256)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            MemoryEfficientSwish()
        )


    def forward(self, background, foreground, mask):
        """
        background               256*64*64   down
        foreground               256*64*64   up
        background_list   bi     256*32*32
        foreground_list   fi     256*32*32
        background_w_list        1024*256*3*3
        foreground_w_list        1024*256*3*3

        background_raw_w_list    1024*256*4*4  重建特征
        foreground_raw_w_list    1024*256*4*4  重建特征
        mm_list                  1024*1*1
        """

        _, _, h, w, = foreground.size()
        mask = F.interpolate(mask, size=[h, w], mode='bilinear', align_corners=True)
        foreground_c = foreground * (1 - mask)
        foreground_m = foreground * mask
        background_c = background * (1 - mask)
        background_m = background * mask

        background_c = self.conv_c(foreground_c, background_c)
        background_m = self.conv_m(foreground_m, background_m)
        background = background_c + background_m
        background = self.conv(background)


        b_shape = list(background.size())  # 256*64*64

        b_list = torch.split(background, 1, dim=0)
        m_list = torch.split(mask, 1, dim=0)

        # 提取 patch background_raw_w_list
        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(background, kernel_size=background_kernel_size,
                                             stride=self.stride * self.rate)
        background_patches = background_patches.view(b_shape[0], -1,
                                                     b_shape[1], background_kernel_size, background_kernel_size)
        background_raw_w_list = torch.split(background_patches, 1, dim=0)

        # 提取 background_list
        background_down = F.interpolate(background, scale_factor=1. / self.rate, mode='bilinear',
                                        align_corners=True)  # 256*32*32
        b_downshape = list(background_down.size())  # 256*32*32
        background_list = torch.split(background_down, 1, dim=0)

        # 提取 background_w_list
        background_pt = extract_patches(background_down, kernel_size=self.kernel_size, stride=self.stride)
        background_pt = background_pt.view(b_downshape[0], -1,
                                           b_downshape[1], self.kernel_size, self.kernel_size)
        background_w_list = torch.split(background_pt, 1, dim=0)


        # process mask
        mask = F.interpolate(mask, size=b_downshape[2:4], mode='bilinear', align_corners=True)

        # 提取mask patch
        m = extract_patches(mask, kernel_size=self.kernel_size, stride=self.stride)
        m = m.view(b_downshape[0], -1, 1, self.kernel_size, self.kernel_size)
        m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
        mm = m.eq(0.).float()
        mm_list = torch.split(mm, 1, dim=0)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1
        escape_NaN = torch.FloatTensor([1e-4])
        if torch.cuda.is_available():
            escape_NaN = escape_NaN.to(mask.device)

        for bi, b_wi, b_raw_wi, mi, background, msk in zip(background_list, background_w_list, background_raw_w_list,
                                                           mm_list, b_list, m_list):
            b_wi = b_wi[0]
            b_wi_normed = b_wi / torch.max(torch.sqrt((b_wi * b_wi).sum([1, 2, 3], keepdim=True)), escape_NaN)

            b_wi_center = b_raw_wi[0]

            # f1
            score_map_f1 = F.conv2d(bi, b_wi_normed, stride=1, padding=padding)
            score_map_f1 = score_map_f1.view(1, b_downshape[2] // self.stride * b_downshape[3] // self.stride,
                                             b_downshape[2], b_downshape[3])
            score_map_f1 = score_map_f1 * mi
            attention_map_f1 = F.softmax(score_map_f1 * self.softmax_scale, dim=1)
            attention_map_f1 = attention_map_f1 * mi
            attention_map_f1 = attention_map_f1.clamp(min=1e-8)
            f1 = F.conv_transpose2d(attention_map_f1, b_wi_center, stride=self.rate, padding=1) / 4.
            f1 = (f1 * msk) + background * (1 - msk)
            output_list.append(f1)

        output = torch.cat(output_list, dim=0)
        return output


class RDCA(nn.Module):

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super(RDCA, self).__init__()

        self.ral = RAL(kernel_size=kernel_size, stride=stride, rate=rate, softmax_scale=softmax_scale)

    def forward(self, background, foreground, mask):
        output = self.ral(background, foreground, mask)

        return output


def test_contextual_attention():
    """Test contextual attention layer with 3-channel image input
  (instead of n-channel feature).
  """
    rate = 2
    stride = 1
    grid = rate * stride
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # b = cv2.imread(args[1])
    b = cv2.imread(r'D:\A_image_inpainting\code\dataset\trian\00000001.jpg')
    b = cv2.resize(b, (b.shape[0] // 4, b.shape[1] // 4))
    # print(args[1])
    h, w, c = b.shape
    b = b[:h // grid * grid, :w // grid * grid, :]
    b = np.transpose(b, [2, 0, 1])
    b = np.expand_dims(b, 0)
    print('Size of imageA: {}'.format(b.shape))

    # f = cv2.imread(args[2])
    f = cv2.imread(r'D:\A_image_inpainting\code\dataset\trian\00000002.jpg')
    f = cv2.resize(f, (f.shape[0] // 4, f.shape[1] // 4))
    h, w, _ = f.shape
    f = f[:h // grid * grid, :w // grid * grid, :]
    f = np.transpose(f, [2, 0, 1])
    f = np.expand_dims(f, 0)
    print('Size of imageB: {}'.format(f.shape))

    bt = torch.Tensor(b)
    ft = torch.Tensor(f)
    atnconv = RDCA(stride=stride)
    yt = atnconv(ft, bt)
    y = yt.cpu().data.numpy().transpose([0, 2, 3, 1])
    outImg = np.clip(y[0], 0, 255).astype(np.uint8)
    plt.imshow(outImg)
    plt.show()
    print(outImg.shape)
    cv2.imwrite('output.jpg', outImg)


if __name__ == '__main__':
    import sys

    # test_contextual_attention(sys.argv)
    test_contextual_attention()
