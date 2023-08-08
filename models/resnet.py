from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from functools import reduce
from torchvision import datasets, models, transforms
#from models.cycle_mlp import CycleBlock
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
#from timm.models import resnet
#from timm.models import resnet
#from models.dla import dla34, dla102
from einops.layers.torch import Rearrange
from models.focal_net import FocalNetBlock
from models.maxvit import MBConv


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)

        return feat


class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        #self.model =models.resnet50(pretrained=True)
        self.feature1 = nn.Sequential(resnet.conv1,
                                  resnet.bn1, resnet.relu,resnet.maxpool)
                                  #resnet.layer1,resnet.layer2,resnet.layer3)
        self.layer1= nn.Sequential(resnet.layer1)
        self.layer2= nn.Sequential(resnet.layer2)
        self.layer3= nn.Sequential(resnet.layer3)
                                  
        self.out_channels = 1024
        
    def forward(self, x):
        feat = self.feature1(x)
        layer1=self.layer1(feat)
        layer2=self.layer2(layer1)
        layer3=self.layer3(layer2)

        return OrderedDict([["feat_res4", layer3]])


    
class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()  # res5
        #self.res5feat=OrderedDict([["layer4", resnet.layer4]])
        #self.layer4 = nn.Sequential(resnet.layer4)
        self.out_channels = [1024, 2048]
        hidden = 256
        output_size = (14,14)
        #self.mlP_model = MLPMixer(in_channels=256, image_size=14, patch_size=1)
        # self.sc_mlp=MLPMixer(
        # input_size = (14,14),
        # patch_size = (1,14),
        # dim = 256)
        #self.simam = SimAM()
        self.focalNet = FocalNetBlock(dim=hidden, input_resolution=196)
        self.norm = nn.BatchNorm2d(hidden)
        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=hidden, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=hidden, out_channels=1024, kernel_size=1)
        #self.mb_conv = MBConv(in_channels=d_model, out_channels=d_model)
        self.patch2vec = nn.Conv2d(1024, hidden, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.vec2patch = Vec2Patch(1024, hidden, output_size, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        
        self.mbconv = MBConv(hidden, hidden)
        self.final_in = nn.Conv2d(in_channels=1024, out_channels=hidden, kernel_size=1)
        self.final_in2 = nn.Conv2d(in_channels=hidden, out_channels=1024, kernel_size=1)
        self.final_out = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        
                
    def forward(self, x):
        input = x
        b, c, h, w = x.size()
        final_in = self.norm(self.final_in(x))
        
        #mbconv=self.mbconv(final_in)
        final_in2 = self.final_in2(final_in)
        
        trans_feat = self.patch2vec(final_in2)

        _, c, h, w = trans_feat.size()
        trans_feat = trans_feat.view(b, c, -1).permute(0, 2, 1)
        
        x_focal_feat=self.focalNet(trans_feat)
        trans_feat = self.vec2patch(x_focal_feat)  + final_in2
        
        final_out = self.final_out(trans_feat)
 
    
        x_feat = F.adaptive_max_pool2d(trans_feat, 1)

        feat = F.adaptive_max_pool2d(final_out, 1)
        trans_features = {}
        trans_features["before_trans"] = x_feat
        trans_features["after_trans"] = feat
        return trans_features
    

def build_resnet(name="resnet50", pretrained=True):
    from torchvision.models import resnet
    resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    #resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    resnet_model = resnet.resnet50(pretrained=True)

    # freeze layers
    resnet_model.conv1.weight.requires_grad_(False)
    resnet_model.bn1.weight.requires_grad_(False)
    resnet_model.bn1.bias.requires_grad_(False)

    return Backbone(resnet_model), Res5Head(resnet_model)
