import torch
from torch import nn
from math import floor

class DNStartBlock(nn.Module):
    def __init__(self, n_inpch, n_otpch):
        super(DNStartBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_otpch)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, n_otpch):
        self.blocks.append(nn.Conv2d(n_inpch, n_otpch,
                kernel_size=7, stride=2, padding=3, bias=False))
        self.blocks.append(nn.BatchNorm2d(n_otpch))
        self.blocks.append(nn.ReLU(True))
        self.blocks.append(nn.MaxPool2d(3, stride=2, padding=1))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class DNCompositeBlock(nn.Module):
    """
    cfg = (# bottleneck output channels, # output channels)
    (4k, k) in original implementation
    no bottleneck if cfg[0] is None
    """
    def __init__(self, n_inpch, cfg):
        super(DNCompositeBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, cfg)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, cfg):
        if cfg[0]:
            self.blocks.append(nn.BatchNorm2d(n_inpch))
            self.blocks.append(nn.ReLU(True))
            self.blocks.append(nn.Conv2d(n_inpch,
                    cfg[0], kernel_size=1, bias=False))
        self.blocks.append(nn.BatchNorm2d(cfg[0]))
        self.blocks.append(nn.ReLU(True))
        self.blocks.append(nn.Conv2d(cfg[0], cfg[1],
                kernel_size=3, padding=1, bias=False))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class DNDenseBlock(nn.Module):
    """
    cfg = list of DNCompositeBlock cfg
    """
    def __init__(self, n_inpch, cfg):
        super(DNDenseBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, cfg)

    def forward(self, x):
        for b in self.blocks:
            x = torch.cat([b(x), x], 1)
        return x

    def make_net(self, n_inpch, cfg):
        for d, c in enumerate(cfg, 1):
            self.blocks.append(DNCompositeBlock(n_inpch, c))
            n_inpch += c[1]
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class DNTransitionBlock(nn.Module):
    def __init__(self, n_inpch, n_otpch):
        super(DNTransitionBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_otpch)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, n_otpch):
        self.blocks.append(nn.BatchNorm2d(n_inpch))
        self.blocks.append(nn.ReLU(True))
        self.blocks.append(nn.Conv2d(n_inpch, n_otpch,
                kernel_size=1, bias=False))
        self.blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class DNClfBlock(nn.Module):
    def __init__(self, n_inpch, n_otp):
        super(DNClfBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_otp)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, n_otp):
        self.blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.blocks.append(nn.Flatten())
        self.blocks.append(nn.Linear(n_inpch, n_otp))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class DenseNet(nn.Module):
    def __init__(self, cfg, n_otp=10, init_weights=False):
        super(DenseNet, self).__init__()
        self.blocks = []
        self.make_net(cfg, n_otp)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return  x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def make_net(self, cfg, n_otp):
        n_inpch = 3
        self.blocks.append(DNStartBlock(n_inpch, cfg['start']))
        n_inpch = cfg['start']
        for c in cfg['dense']:
            self.blocks.append(DNDenseBlock(n_inpch, c))
            n_inpch += sum(n_ch for _, n_ch in c)
            n_otpch = floor(n_inpch * cfg['compress'])
            self.blocks.append(DNTransitionBlock(n_inpch, n_otpch))
            n_inpch = n_otpch
        self.blocks.append(DNClfBlock(n_inpch, n_otp))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)