import torch
from torch import nn

class InV1StartBlock(nn.Module):
    def __init__(self, n_inpch=3):
        super(InV1StartBlock, self).__init__()
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
        self.blocks.append(nn.MaxPool2d(3, stride=2, ceil_mode=True))
        self.blocks.append(nn.Conv2d(64, 64, kernel_size=1, bias=False))
        self.blocks.append(nn.BatchNorm2d(64, eps=0.001))
        self.blocks.append(nn.Conv2d(64, 192,
                kernel_size=3, padding=1, bias=False))
        self.blocks.append(nn.BatchNorm2d(192, eps=0.001))
        self.blocks.append(nn.MaxPool2d(3, stride=2, ceil_mode=True))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class InV1InConvBlock(nn.Module):
    """
    n_redch = 0 if dimension reduction not done
    """
    def __init__(self, n_inpch, ksize, n_redch, n_otpch):
        super(InV1InConvBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, ksize, n_redch, n_otpch)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x
    
    def make_net(self, n_inpch, ksize, n_redch, n_otpch):
        if n_redch:
            self.blocks.append(nn.Conv2d(n_inpch, n_redch, kernel_size=1))
            self.blocks.append(nn.BatchNorm2d(n_redch))
            n_inpch = n_redch
        self.blocks.append(nn.Conv2d(n_inpch, n_otpch,
                kernel_size=ksize, padding=(ksize // 2)))
        self.blocks.append(nn.BatchNorm2d(n_otpch))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class InV1InPoolBlock(nn.Module):
    """
    n_redch = 0 if dimension reduction not done
    """
    def __init__(self, n_inpch, n_redch):
        super(InV1InPoolBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_redch)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x
    
    def make_net(self, n_inpch, n_redch):
        self.blocks.append(nn.MaxPool2d(kernel_size=3,
                stride=1, padding=1, ceil_mode=True))
        if n_redch:
            self.blocks.append(nn.Conv2d(n_inpch, n_redch, kernel_size=1))
            self.blocks.append(nn.BatchNorm2d(n_redch))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class InV1InceptBlock(nn.Module):
    """
    cfg = {
            1: (n_redch1=0, n_otpch1),
            3: (n_redch3, n_otpch3),
            5: (n_redch5, n_otpch5),
            'm': n_redchm
        }
    """
    def __init__(self, n_inpch, cfg, mp_cfg):
        super(InV1InceptBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, cfg, mp_cfg)

    def forward(self, x):
        x = torch.cat([b(x) for b in self.blocks[0]], 1)
        if len(self.blocks) == 2:
            x = self.blocks[1](x)
        return x
    
    def make_net(self, n_inpch, cfg, mp_cfg):
        row = []
        for ksize in (1, 3, 5):
            row.append(InV1InConvBlock(n_inpch,
                    ksize, cfg[ksize][0], cfg[ksize][1]))
        row.append(InV1InPoolBlock(n_inpch, cfg['m']))
        self.blocks.append(row)
        if mp_cfg[0]:
            self.blocks.append(
                    nn.MaxPool2d(mp_cfg[1], mp_cfg[2], ceil_mode=True))
        for i, b in enumerate(self.blocks[0]):
            self.add_module('B_001_' + str(i).zfill(3), b)
        if len(self.blocks) == 2:
            self.add_module('B_002', self.blocks[1])

class InV1ClfBlock(nn.Module):
    def __init__(self, n_inpch, n_otp, cfg):
        super(InV1ClfBlock, self).__init__()
        self.blocks = []
        self.make_net(n_inpch, n_otp, cfg)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def make_net(self, n_inpch, n_otp, cfg):
        if cfg == 'aux':
            self.blocks.append(nn.AdaptiveAvgPool2d((4, 4)))
            self.blocks.append(nn.Conv2d(n_inpch, 128, kernel_size=1))
            self.blocks.append(nn.BatchNorm2d(128))
            self.blocks.append(nn.Flatten())
            self.blocks.append(nn.Linear(2048, 1024))
            self.blocks.append(nn.ReLU(True))
            self.blocks.append(nn.Dropout(0.7))
            self.blocks.append(nn.Linear(1024, n_otp))
        else:
            self.blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.blocks.append(nn.Flatten())
            self.blocks.append(nn.Dropout(0.2))
            self.blocks.append(nn.Linear(n_inpch, n_otp))
        for i, b in enumerate(self.blocks):
            self.add_module('B_' + str(i).zfill(3), b)

class InceptionV1(nn.Module):
    def __init__(self, cfg, n_otp=10, init_weights=False):
        super(InceptionV1, self).__init__()
        self.blocks = []
        self.make_net(cfg, n_otp)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        otp = []
        for r in self.blocks:
            if ('aux' in r.keys()) and self.training:
                otp.append(r['aux'](x))
            elif 'final' in r.keys():
                otp.append(r['final'](x))
            if 'start' in r.keys():
                x = r['start'](x)
            elif 'incept' in r.keys():
                x = r['incept'](x)
        return torch.cat(otp, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()),
                        dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_net(self, cfg, n_otp):
        n_inpch = 3
        self.blocks.append({'start': InV1StartBlock(n_inpch)})
        n_inpch = 192
        for in_cfg, mp_cfg, clf_cfg in cfg:
            row = {}
            row['incept'] = InV1InceptBlock(n_inpch, in_cfg, mp_cfg)
            if clf_cfg:
                row[clf_cfg] = InV1ClfBlock(n_inpch, n_otp, clf_cfg)
            self.blocks.append(row)
            n_inpch = sum([in_cfg[i][1] for i in (1, 3, 5)]) + in_cfg['m']
        for i, r in enumerate(self.blocks):
            for j, btype in enumerate(r.keys()):
                self.add_module('B_' + str(i).zfill(3) + '_' + str(j).zfill(3),
                        r[btype])