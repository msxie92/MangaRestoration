import os
import math
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import functools
import torch.nn.functional as F
from core.spectral_norm import use_spectral_norm

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    elif layer_type == 'selu':
        nl_layer = functools.partial(nn.SELU, inplace=True)
    elif layer_type == 'prelu':
        nl_layer = functools.partial(nn.PReLU)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class LayerNormWrapper(nn.Module):
    def __init__(self, num_features):
        super(LayerNormWrapper, self).__init__()
        self.num_features = int(num_features)

    def forward(self, x):
        #y = x.to(torch.float32)
        #x = nn.LayerNorm([self.num_features, x.size()[2], x.size()[3]]).cuda()(y).to(x.dtype)
        x = nn.LayerNorm([self.num_features, x.size()[2], x.size()[3]], elementwise_affine=False).cuda()(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False, padding_mode='reflect')
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False, padding_mode='reflect')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, nl_layer=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nl_layer(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        # self.downsample = downsample
        self.stride = stride
        if stride != 1:
            self.downsample = convMeanpool(inplanes, planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.stride != 1:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True, padding_mode='reflect')

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True, padding_mode='reflect')]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.randn(channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x, noise, mask):
        W,_ = torch.split(self.weight.view(1, -1, 1, 1), self.channels // 2, dim=1)
        B,_ = torch.split(self.bias.view(1, -1, 1, 1), self.channels // 2, dim=1)
        Z = torch.zeros_like(W)
        w = torch.cat([W,Z], dim=1).to(x.device)
        b = torch.cat([B,Z], dim=1).to(x.device)
        # adds = w * noise.to(x.device) + b
        # return x + adds.type_as(x)*(1-mask)
        adds = w * torch.randn_like(x) + b
        mask = F.interpolate(mask, (x.shape[2],x.shape[3]))
        return x + adds.type_as(x)*(1-mask)

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        norm_layer = functools.partial(LayerNormWrapper)
        nl_layer = get_non_linearity('lrelu')
        self.bn1 = norm_layer(input_channels)
        self.relu = nl_layer(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1, 1, bias = False, padding_mode='reflect')
        self.bn2 = norm_layer(output_channels//4)
        self.relu = nl_layer(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride, padding = 1, bias = False, padding_mode='reflect')
        self.bn3 = norm_layer(output_channels//4)
        self.relu = nl_layer(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = False, padding_mode='reflect')
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False, padding_mode='reflect')
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class AttentionModule_stage0(nn.Module):
    # input size is 112*112
    def __init__(self, in_channels, out_channels, size1=(112, 112), size2=(56, 56), size3=(28, 28), size4=(14, 14)):
        super(AttentionModule_stage0, self).__init__()
        norm_layer = functools.partial(LayerNormWrapper)
        nl_layer = get_non_linearity('lrelu')
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 56*56
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 28*28
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 14*14
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.skip3_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 7*7
        self.softmax4_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        # self.interpolation4 = nn.UpsamplingBilinear2d(size=size4)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        # self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax6_blocks = ResidualBlock(in_channels, out_channels)
        # self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax7_blocks = ResidualBlock(in_channels, out_channels)
        # self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax8_blocks = nn.Sequential(
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False, padding_mode='reflect'),
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size=1, stride=1, bias = False, padding_mode='reflect'),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bicubic') + y

    def forward(self, x):
        # 112*112
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        # 56*56
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        # 28*28
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        # 14*14
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
        out_mpool4 = self.mpool4(out_softmax3)
        # 7*7
        out_softmax4 = self.softmax4_blocks(out_mpool4)
        out_interp4 = self._upsample_add(out_softmax4, out_softmax3)
        out = out_interp4 + out_skip3_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp3 = self._upsample_add(out_softmax5, out_softmax2)
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax6 = self.softmax6_blocks(out)
        out_interp2 = self._upsample_add(out_softmax6, out_softmax1)
        out = out_interp2 + out_skip1_connection
        out_softmax7 = self.softmax7_blocks(out)
        out_interp1 = self._upsample_add(out_softmax7, out_trunk)
        out_softmax8 = self.softmax8_blocks(out_interp1)
        out = (1 + out_softmax8) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule_stage1(nn.Module):
    # input size is 56*56
    def __init__(self, in_channels, out_channels, size1=(56, 56), size2=(28, 28), size3=(14, 14)):
        super(AttentionModule_stage1, self).__init__()
        norm_layer = functools.partial(LayerNormWrapper)
        nl_layer = get_non_linearity('lrelu')
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        # self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        # self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        # self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False, padding_mode='reflect'),
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False, padding_mode='reflect'),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bicubic') + y

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #
        out_interp3 = self._upsample_add(out_softmax3, out_softmax2)
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self._upsample_add(out_softmax4, out_softmax1)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self._upsample_add(out_softmax5, out_trunk)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule_stage2(nn.Module):
    # input image size is 28*28
    def __init__(self, in_channels, out_channels, size1=(28, 28), size2=(14, 14)):
        super(AttentionModule_stage2, self).__init__()
        norm_layer = functools.partial(LayerNormWrapper)
        nl_layer = get_non_linearity('lrelu')
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        # self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        # self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax4_blocks = nn.Sequential(
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding_mode='reflect'),
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding_mode='reflect'),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bicubic') + y

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = self._upsample_add(out_softmax2, out_softmax1)

        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self._upsample_add(out_softmax3, out_trunk)
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule_stage3(nn.Module):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels):
        super(AttentionModule_stage3, self).__init__()
        norm_layer = functools.partial(LayerNormWrapper)
        nl_layer = get_non_linearity('lrelu')
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        # self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding_mode='reflect'),
            norm_layer(out_channels),
            nl_layer(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding_mode='reflect'),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bicubic') + y

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = self._upsample_add(out_softmax1, out_trunk)
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last, out_softmax2



class ScaleEstimator(BaseNetwork):
  def __init__(self, input_nc=1, output_nc=1, ndf=64, n_blocks=4, 
               norm_layer=functools.partial(LayerNormWrapper), nl_layer=nn.ReLU, init_weights=True):
    super(ScaleEstimator, self).__init__()
    max_ndf = 4
    conv_layers = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True, padding_mode='reflect')]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)
      output_ndf = ndf * min(max_ndf, n + 1)
      conv_layers += [BasicBlock(input_ndf, output_ndf, 2, norm_layer, nl_layer)]
    conv_layers += [nl_layer()]
    self.conv = nn.Sequential(*conv_layers)
    self.pool = nn.Sequential(*[nn.AdaptiveAvgPool2d(1)])
    self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.n_blocks = n_blocks

    if init_weights:
      self.init_weights()

  def forward(self, x):
    x_conv = self.conv(x)
    x_conv = self.pool(x_conv)#*mask
    conv_flat = x_conv.view(x.size(0), -1)
    output = self.fc(conv_flat)
    return output

class MangaRestorator(BaseNetwork):

    def __init__(self, input_nc=1, output_nc=1, ndf=32, bilinear=True,
               norm_layer=functools.partial(LayerNormWrapper), nl_layer=nn.ReLU, init_weights=True):
        super(MangaRestorator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1, padding=0, bias=False, padding_mode='reflect'),
            norm_layer(ndf),
            nl_layer(inplace=True)
        ) 
        self.residual_block1 = ResidualBlock(ndf, ndf)  
        self.attention_module1 = AttentionModule_stage3(ndf, ndf) 
        self.residual_block2 = ResidualBlock(ndf, ndf)  
        self.noise_block2 = ApplyNoise(ndf)
        self.attention_module2 = AttentionModule_stage2(ndf, ndf)  
        self.attention_module2_2 = AttentionModule_stage2(ndf, ndf)  
        self.residual_block3 = ResidualBlock(ndf, ndf)  
        self.noise_block3 = ApplyNoise(ndf)
        self.attention_module3 = AttentionModule_stage1(ndf, ndf)  
        self.attention_module3_2 = AttentionModule_stage1(ndf, ndf)  

        self.noise_block4 = ApplyNoise(ndf)
        self.residual_block4 = ResidualBlock(ndf, ndf)  
        self.residual_block5 = ResidualBlock(ndf, ndf)  

        self.residual_block0 = ResidualBlock(ndf, ndf)  
        self.final_atten = nn.Conv2d(ndf,1,1,1,0)
        self.final = nn.Conv2d(ndf,output_nc,1,1,0)
        self.mask = nn.Sequential(
            # nn.Conv2d(ndf, ndf, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(ndf, 9, 1, padding=0))
            use_spectral_norm(nn.Conv2d(ndf, ndf, 3, padding=1, padding_mode='reflect')),
            nn.ReLU(inplace=True),
            use_spectral_norm(nn.Conv2d(ndf, 9, 1, padding=0)))

        self.noise_inputs = []
        shape = [1, 1, ndf, ndf]
        for i in range(4):
            self.noise_inputs.append(torch.randn(*shape))

        self.bilinear = bilinear

        if init_weights:
            self.init_weights()

    def upsample_bilinear(self,x,scale):
        scale_int = math.ceil(scale)
        N,C,H,W = x.size()
        x = F.interpolate(x, size=(round(H*scale), round(W*scale)), mode='bicubic')
        _,_,H2,W2 = x.size()
        # x = torch.stack([x]*2,2)
        # x = torch.stack([x]*2,4)
        # x = x.view(N,C,H2,1,W2,1)
        x = torch.cat([x]*2,2)
        x = torch.cat([x]*2,3)

        return x.contiguous().view(N,C,H2*2,W2*2)[:,:,:(H*scale_int), :(W*scale_int)]

    def upsample_convex(self,x,scale):
        scale_int = math.ceil(scale)
        N,C,H,W = x.size()
        up_x = self.upsample_bilinear(x, scale)
        cols = nn.functional.unfold(up_x, 3, padding=1)  ### (N*r*r,inC*ks*ks,inH*inW)
        cols = cols.view(N, C, -1, scale_int*H, scale_int*W)
        coef = self.mask(up_x) ### (N,ks*ks,outH, outW)
        coef = coef.view(N, 1, -1, scale_int*H, scale_int*W)#.permute(0,1,2,3,5,4,6).contiguous() ### (ks*ks,r,r,inH,inW)
        coef = torch.softmax(coef, dim=2) ### (N,1,ks*ks,r,r,inH,inW)
        
        out = torch.sum(cols*coef, dim=2)#.permute(0, 1, 2, 4, 3, 5) ### (N,inC,r,inH,r,inW)
        return out

    def forward(self, x, scale):
        _,_,h,w = x.shape
        out = self.conv1(x)
        out = self.residual_block1(out)
        out, atten = self.attention_module1(out)
        atten = torch.sigmoid(self.final_atten(self.residual_block0(atten)))
        
        out = self.residual_block2(out)
        # out = self.noise_block2(out, self.noise_inputs[1], atten.detach())
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)

        atten = self.upsample_bilinear(atten, scale)
        if self.bilinear:
            out = self.upsample_bilinear(out, scale)
        else:
            out = self.upsample_convex(out, scale)
        out = self.residual_block3(out)
        out = self.noise_block3(out, self.noise_inputs[2], atten.detach())
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)

        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.final(out)

        return out, atten

