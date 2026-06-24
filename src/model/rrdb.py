import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        return x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class ResidualDenseBlock_5C_TL(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, out_text_channels=32):
        super(ResidualDenseBlock_5C_TL, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc + out_text_channels, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, text_emb):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4, text_emb), 1))

        return x5 * 0.166 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDB_TL(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, out_text_channels=32):
        super(RRDB_TL, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_TL(nf, gc, out_text_channels=out_text_channels)
        self.RDB2 = ResidualDenseBlock_5C_TL(nf, gc, out_text_channels=out_text_channels)
        self.RDB3 = ResidualDenseBlock_5C_TL(nf, gc, out_text_channels=out_text_channels)

    def forward(self, x_in):

        (x, text_emb) = x_in

        out = self.RDB1(x, text_emb)
        out = self.RDB2(out, text_emb)
        out = self.RDB3(out, text_emb)
        return out * 0.2 + x


class RRDB_TL_LP(nn.Module):
    """
    RRDB backbone + Text Label conditioning for License Plate SR (2x scale).

    - InfoGen integrated (like TSRN_TL) — forward(x, text_emb) receives label_vecs raw
    - 2x Pixel Shuffle upsampling instead of 3x interpolate
    - ModuleList for RRDB_TL blocks (fix bug tuple through nn.Sequential)
    - Output tanh in [-1,1]
    """
    def __init__(self, scale_factor=2, width=80, height=32,
                 in_nc=3, nf=64, nb=8, gc=32,
                 text_emb=36, out_text_channels=64):
        super(RRDB_TL_LP, self).__init__()
        assert math.log(scale_factor, 2) % 1 == 0

        self.text_emb_ch  = text_emb
        self.scale_factor = scale_factor

        # InfoGen (same as TSRN_TL)
        self.infoGen = InfoGen(text_emb, out_text_channels)

        # Feature extraction
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # RRDB_TL blocks (ModuleList instead of nn.Sequential)
        self.rrdb_blocks = nn.ModuleList(
            [RRDB_TL(nf=nf, gc=gc, out_text_channels=out_text_channels)
             for _ in range(nb)]
        )
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 2x Pixel Shuffle upsampling
        self.upconv    = nn.Conv2d(nf, nf * (scale_factor ** 2), 3, 1, 1, bias=True)
        self.ps        = nn.PixelShuffle(scale_factor)
        self.HRconv    = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, in_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, text_emb=None):

        if text_emb is None:
            N, _, H, W = x.shape
            text_emb = torch.zeros(N, self.text_emb_ch, 1, 26, device=x.device)

        # InfoGen -> spatial text features, resize to LR spatial size
        spatial_t_emb = self.infoGen(text_emb)
        spatial_t_emb = F.interpolate(
            spatial_t_emb, (x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=True
        )

        # Feature extraction
        fea = self.conv_first(x)

        # RRDB_TL blocks
        out = fea
        for block in self.rrdb_blocks:
            out = block((out, spatial_t_emb))
        fea = fea + self.trunk_conv(out)

        # 2x Pixel Shuffle
        fea = self.lrelu(self.ps(self.upconv(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return torch.tanh(out)
