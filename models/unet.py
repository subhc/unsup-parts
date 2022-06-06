import logging
import torch
import torch.nn as nn
import torch.nn.functional as F



class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = UNet.make_double_conv(in_ch, out_ch, batch_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_layers, batch_norm=False):
        super(UNet, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_layers = n_layers

        nc = 64
        self.batch_norm = batch_norm
        self.inc = self.make_double_conv(n_in_channels, nc, batch_norm)

        self.down_layers = []
        for i in range(n_layers - 1):
            self.down_layers.append(self._make_down_layer(nc, 2 * nc))
            self.add_module('down{}'.format(i + 1), self.down_layers[-1])
            nc *= 2

        self.down_layers.append(self._make_down_layer(nc, nc))
        self.add_module('down{}'.format(n_layers), self.down_layers[-1])

        self.up_layers = []
        for i in range(n_layers - 1):
            self.up_layers.append(UpLayer(nc * 2, nc // 2, self.batch_norm))
            self.add_module('up{}'.format(i + 1), self.up_layers[-1])
            nc = nc // 2

        self.up_layers.append(UpLayer(nc * 2, nc, self.batch_norm))
        self.add_module('up{}'.format(n_layers), self.up_layers[-1])

        self.final_conv = nn.Conv2d(nc, n_out_channels, 1)

        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.info(f'Network:\n'
                     f'\t{self.n_in_channels} input channels\n'
                     f'\t{self.n_out_channels} output channels \n'
                     f'\t{self.n_layers} layers\n'
                     f'\t{"batch_norm" if self.batch_norm else "no batch_norm"}')

    @staticmethod
    def make_double_conv(in_ch, out_ch, batch_norm):
        modules = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)]
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_ch))
        modules.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        modules.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False))
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_ch))
        modules.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        return nn.Sequential(*modules)

    def _make_down_layer(self, in_ch, out_ch):
        return nn.Sequential(nn.MaxPool2d(2), self.make_double_conv(in_ch, out_ch, self.batch_norm))

    def forward(self, x):
        xs = [self.inc(x)]
        for dl in self.down_layers:
            xs.append(dl(xs[-1]))
        x = xs[-1]
        for i, ul in enumerate(self.up_layers):
            x = ul(x, xs[-i - 2])
        x = self.final_conv(x)

        return None, None, x, None

    def optim_parameters(self, args):
        return [{'params': self.parameters(), 'lr': 10 * args.learning_rate}]

