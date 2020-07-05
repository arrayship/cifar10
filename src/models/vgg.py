from torch import nn

class VGGConvBlock(nn.Module):
    def __init__(self, n_inpch, cfg, batch_norm=False):
        super(VGGConvBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, cfg, batch_norm)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x
    
    def make_net(self, n_inpch, cfg, batch_norm):
        for v in cfg:
            self.blocks.append(nn.Conv2d(n_inpch, v, kernel_size=3, padding=1))
            if batch_norm:
                self.blocks.append(nn.BatchNorm2d(v))
            self.blocks.append(nn.ReLU(inplace=True))
            n_inpch = cfg[- 1]
        self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class VGGClfBlock(nn.Module):
    def __init__(self, n_inpch, n_otp):
        super(VGGClfBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_otp)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, n_otp):
        self.blocks.append(nn.AdaptiveAvgPool2d((7, 7)))
        self.blocks.append(nn.Flatten())
        self.blocks.append(nn.Linear(n_inpch * 7 * 7, 4096))
        self.blocks.append(nn.ReLU(True))
        self.blocks.append(nn.Dropout())
        self.blocks.append(nn.Linear(4096, 4096))
        self.blocks.append(nn.ReLU(True))
        self.blocks.append(nn.Dropout())
        self.blocks.append(nn.Linear(4096, n_otp))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class VGG(nn.Module):
    def __init__(self, cfg, n_otp=10, init_weights=False):
        super(VGG, self).__init__()
        self.blocks = []
        self.make_net(cfg['cfg'], n_otp, cfg['batch_norm'])
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_net(self, cfg, n_otp, batch_norm=False):
        n_inpch = self.make_conv(cfg, batch_norm)
        self.make_clf(n_inpch, n_otp)
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

    def make_conv(self, cfg, batch_norm=False):
        n_inpch = 3
        for c in cfg:
            self.blocks.append(VGGConvBlock(n_inpch, c, batch_norm))
            n_inpch = c[- 1]
        return n_inpch
        
    def make_clf(self, n_inpch, n_otp):
        self.blocks.append(VGGClfBlock(n_inpch, n_otp))