import torch
from torch import nn

class RNStartBlock(nn.Module):
    def __init__(self, n_inpch=3):
        super(RNStartBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch):
        self.blocks.append(nn.Conv2d(n_inpch, 64,
                kernel_size=7, stride=2, padding=3, bias=False))
        self.blocks.append(nn.BatchNorm2d(64, eps=0.001))
        self.blocks.append(nn.ReLU(True))
        self.blocks.append(nn.MaxPool2d(3, stride=2, padding=1))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class RNIdBlock(nn.Module):
    def __init__(self, n_inpch, n_otpch):
        super(RNIdBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_otpch)
    
    def forward(self, x):
        return self.blocks[0](x)

    def make_net(self, n_inpch, n_otpch):
        if n_inpch == n_otpch:
            self.blocks.append(nn.Identity())
        else:
            self.blocks.append(nn.Conv2d(n_inpch, n_otpch, kernel_size=1))
            self.blocks.append(nn.BatchNorm2d(n_otpch))
        self.add_module('B_000', self.blocks[0])

class RNConvBlock(nn.Module):
    """
    cfg = [(ksize1, n_otpch1), (ksize2, n_otpch2), (ksize3, n_otpch3)]
    """
    def __init__(self, n_inpch, cfg):
        super(RNConvBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, cfg)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, cfg):
        for i, (ksize, n_otpch) in enumerate(cfg):
            self.blocks.append(nn.Conv2d(n_inpch, n_otpch,
                    kernel_size=ksize, padding=(ksize // 2)))
            self.blocks.append(nn.BatchNorm2d(n_otpch))
            if i < len(cfg) - 1:
                self.blocks.append(nn.ReLU(True))
            n_inpch = n_otpch
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class RNResBlock(nn.Module): # stride setting needed # dimension increase
    """
    cfg = [(ksize1, n_otpch1), (ksize2, n_otpch2), (ksize3, n_otpch3)]
    """
    def __init__(self, n_inpch, cfg):
        super(RNResBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, cfg)

    def forward(self, x):
        x = self.blocks[0][0](x) + self.blocks[0][1](x)
        x = self.blocks[1](x)
        return x

    def make_net(self, n_inpch, cfg):
        row = []
        row.append(RNIdBlock(n_inpch, cfg[- 1][1]))
        row.append(RNConvBlock(n_inpch, cfg))
        self.blocks.append(row)
        self.blocks.append(nn.ReLU(True))
        for i, b in enumerate(self.blocks[0]):
            self.add_module('B_000_' + str(i).zfill(3), b)
        self.add_module('B_001', self.blocks[1])
            
class RNClfBlock(nn.Module):
    def __init__(self, n_inpch, n_otp):
        super(RNClfBlock, self).__init__()
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

class ResNet(nn.Module):
    def __init__(self, cfg, n_otp=10, init_weights=False):
        super(ResNet, self).__init__()
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_net(self, cfg, n_otp):
        n_inpch = 3
        self.blocks.append(RNStartBlock(n_inpch))
        n_inpch = 64
        for c in cfg:
            self.blocks.append(RNResBlock(n_inpch, c))
            n_inpch = c[- 1][1]
        self.blocks.append(RNClfBlock(n_inpch, n_otp))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)