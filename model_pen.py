# model_pen.py
# Probability Estimation Network (PEN) - Standard UNet
# Output: 1-channel rain-probability map in [0,1]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleConv, self).__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv. Uses ConvTranspose2d for upsample."""
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()

        # if bilinear, use nn.Upsample + conv, else ConvTranspose2d
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            # in_ch is usually doubled due to concatenation of skip connection
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        x1: coming from lower layer (to be upsampled)
        x2: skip connection from encoder
        """
        if isinstance(self.up, nn.Upsample):
            x1 = self.up(x1)
        else:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad x1 to have same size as x2 (in case of odd sizes)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 conv -> sigmoid"""
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetPEN(nn.Module):
    """
    Standard UNet used for PEN.
    Input: RGB image (3 channels)
    Output: 1-channel probability map (use sigmoid)
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, bilinear=False):
        super(UNetPEN, self).__init__()
        filters = base_filters

        self.inc = DoubleConv(in_channels, filters)
        self.down1 = Down(filters, filters * 2)
        self.down2 = Down(filters * 2, filters * 4)
        self.down3 = Down(filters * 4, filters * 8)
        self.down4 = Down(filters * 8, filters * 8)  # bottom-most (no further doubling)

        self.up1 = Up(filters * 16, filters * 4, bilinear=bilinear)
        self.up2 = Up(filters * 8, filters * 2, bilinear=bilinear)
        self.up3 = Up(filters * 4, filters * 1, bilinear=bilinear)
        self.up4 = Up(filters * 2, filters, bilinear=bilinear)

        self.outc = OutConv(filters, out_channels)
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # [B, f, H, W]
        x2 = self.down1(x1)   # [B, 2f, H/2, W/2]
        x3 = self.down2(x2)   # [B, 4f, H/4, W/4]
        x4 = self.down3(x3)   # [B, 8f, H/8, W/8]
        x5 = self.down4(x4)   # [B, 8f, H/16, W/16] (bottleneck)

        # Decoder (note Up expects (x1_from_below, x_skip))
        # up1: combine x5 and x4
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        out = self.outc(u4)
        prob_map = self.sigmoid(out)
        return prob_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # quick sanity check: ensures the model builds and outputs expected shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetPEN(in_channels=3, out_channels=1, base_filters=64).to(device)
    x = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)   # should be [2,1,128,128]
