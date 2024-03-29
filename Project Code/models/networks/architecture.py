"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, feature_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        # FEATURE: Replace opt.semantic_nc with feature_nc
        self.norm_0 = SPADE(spade_config_str, fin, feature_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, feature_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, feature_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    # FEATURE: the semantic segmentation map "seg" is replaced with the feature map "feature"
    def forward(self, x, feature):
        x_s = self.shortcut(x, feature)

        dx = self.conv_0(self.actvn(self.norm_0(x, feature)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, feature)))

        out = x_s + dx

        return out

    def shortcut(self, x, feature):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, feature))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
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
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# VGG architecter, used for creating feature maps for SPADE blocks.
class VGG19Features(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 16):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 21):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 25):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        for x in range(25, 30):
            self.slice7.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.feature_nc = [64, 128, 256, 256, 512, 512, 512]

    def forward(self, X):
        h_relu1 = self.slice1(X)  # 3 -> 64
        h_relu2 = self.slice2(h_relu1)  # 64 -> 128
        h_relu3 = self.slice3(h_relu2)  # 128 -> 256
        h_relu4 = self.slice4(h_relu3)  # 256 -> 256
        h_relu5 = self.slice5(h_relu4)  # 256 -> 512
        h_relu6 = self.slice6(h_relu5)  # 512 -> 512
        h_relu7 = self.slice7(h_relu6)  # 512 -> 512
        
		#h_relu1 = torch.empty(h_relu1.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu1
        #h_relu2 = torch.empty(h_relu2.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu2
        #h_relu3 = torch.empty(h_relu3.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu3
        #h_relu4 = torch.empty(h_relu4.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu4
        #h_relu5 = torch.empty(h_relu5.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu5
        #h_relu6 = torch.empty(h_relu6.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu6
        #h_relu7 = torch.empty(h_relu7.size(), device = 'cuda').normal_(mean=0.0, std=0.5) + h_relu7
		
        out = [h_relu1.detach(), h_relu2.detach(), h_relu3.detach(), h_relu4.detach(), h_relu5.detach(), h_relu6.detach(), h_relu7.detach()]
        return out


# VGG architecter, used for the style and content loss using a pretrained VGG network
class VGG19StyleAndContent(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1):  # style layer 0: conv1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):  # style layer 5: conv2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):  # style layer 10: conv3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):  # style layer 19: conv4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 22):  # content layer 21: conv4_2
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29):  # style layer 28: conv5_1
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        style = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu6]
        content = h_relu5
        return style, content
