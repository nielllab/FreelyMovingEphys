"""
core.py

notes:
took out skip feature, took out laplace
"""
from collections import OrderedDict, defaultdict
from itertools import count
from warnings import warn
from torch import nn
from torch.nn.init import xavier_normal
import torch
from torch.nn import functional as F, Parameter
import torch.nn.init as init
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_out_dims(inshape,Cout,dialation=1,ksize=1,stride=1,padding=1):# Cout = num channels out
        if type(ksize) == int:
            ksize = (ksize,ksize,ksize)
        if type(dialation) == int:
            dialation = (dialation,dialation,dialation)
        if type(stride) == int:
            stride = (stride,stride,stride)
        if type(padding) == int:
            padding = (padding,padding,padding)
        C,D,H,W = inshape
        Dout = int(((D + 2*padding[0] - dialation[0]*(ksize[0]-1)-1)/stride[0]) + 1)
        Hout = int(((H + 2*padding[1] - dialation[1]*(ksize[1]-1)-1)/stride[1]) + 1)
        Wout = int(((W + 2*padding[2] - dialation[2]*(ksize[2]-1)-1)/stride[2]) + 1)
        return [Cout,Dout,Hout,Wout] # ch, depth, height, width

class Core:
    def initialize(self):
        log.info('Not initializing anything')

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: 'gamma' in x or 'skip' in x, dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'

class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        if cuda:
            self = self.cuda()
    
    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.fill_(0)

class Stacked2dCore(Core2d, nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers=3,
                 gamma_hidden=0, gamma_input=0., final_nonlinearity=True, bias=False,
                 momentum=0.1, pad_input=True, batch_norm=True, **kwargs):
        # log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.features = nn.Sequential()

        # first layer
        layer = OrderedDict()
        layer['conv'] = \
            nn.Conv2d(input_channels, hidden_channels, input_kern,
                      padding=input_kern // 2 if pad_input else 0, bias=bias)
        if batch_norm:
            layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if (layers > 1 or final_nonlinearity):
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer['conv'] = \
                nn.Conv2d(hidden_channels,
                          hidden_channels, hidden_kern,
                          padding=hidden_kern // 2, bias=bias)
            if batch_norm:
                layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if (final_nonlinearity or l < self.layers - 1):
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

        self.features.add_module('layerfc', nn.Linear(2*128*128,12)) # fully connected layer

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for name, feat in self.features.named_children():
            if name == 'layerfc':
                input_ = input_.flatten(start_dim=1)
            input_ = feat(input_)
            # ret.append(input_)
        # return torch.cat(ret, dim=1)
        return input_

class Stacked3dCore(Core2d, nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, num_units, layers=3,
                 gamma_hidden=0, gamma_input=0., final_nonlinearity=True, bias=False,
                 momentum=0.1, pad_input=True, batch_norm=True, img_resize=64, spike_history_len=4, stride=(1,2,2), **kwargs):  # stride: time/depth, height, width
        # log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        inshape = [input_channels, spike_history_len, img_resize, img_resize]

        self.features = nn.Sequential()

        # first layer
        layer = OrderedDict()
        layer['conv'] = \
            nn.Conv3d(input_channels, hidden_channels, input_kern,
                      padding=input_kern // 2, bias=bias, stride=stride)
        if batch_norm:
            layer['norm'] = nn.BatchNorm3d(hidden_channels, momentum=momentum)
        if (layers > 1 or final_nonlinearity):
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer['conv'] = \
                nn.Conv3d(hidden_channels,
                          hidden_channels, hidden_kern,
                          padding=hidden_kern // 2, bias=bias, stride=stride)
            if batch_norm:
                layer['norm'] = nn.BatchNorm3d(hidden_channels, momentum=momentum)
            if (final_nonlinearity or l < self.layers - 1):
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

        for i in range(layers):
            Cout,Dout,Hout,Wout = calc_out_dims(inshape, hidden_channels, ksize=input_kern, padding=hidden_kern // 2, stride=stride)
            inshape = [Cout,Dout,Hout,Wout]
            input_kern = hidden_kern
        
        self.features.add_module('layerfc', nn.Linear(Cout*Dout*Hout*Wout,num_units)) # fully connected layer: 2*128*128,12

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for name, feat in self.features.named_children():
            if name == 'layerfc':
                input_ = input_.flatten(start_dim=1)
            input_ = feat(input_)
            # ret.append(input_)
        # return torch.cat(ret, dim=1)
        return input_