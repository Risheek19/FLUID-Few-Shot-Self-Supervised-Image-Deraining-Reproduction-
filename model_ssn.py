import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=1):
        super().__init__()
        self.down_conv = ConvBlock(in_ch, out_ch, stride=2)
        self.res_blocks = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(n_blocks)])

    def forward(self, x):
        x = self.down_conv(x)
        x = self.res_blocks(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.up.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.reduce_conv = None
        self.res_blocks = None

    def set_res_blocks(self, concat_ch, out_ch, n_blocks=1):
        self.reduce_conv = ConvBlock(concat_ch, out_ch)
        self.res_blocks = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(n_blocks)])

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.reduce_conv(x)
        x = self.res_blocks(x)
        return x


class SSNResidualUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_filters=64, n_down=4, res_blocks_per_stage=2):
        super().__init__()
        f = base_filters

        self.inc   = ConvBlock(in_channels, f)
        self.down1 = Down(f, f * 2, n_blocks=res_blocks_per_stage)
        self.down2 = Down(f * 2, f * 4, n_blocks=res_blocks_per_stage)
        self.down3 = Down(f * 4, f * 8, n_blocks=res_blocks_per_stage)
        self.down4 = Down(f * 8, f * 8, n_blocks=res_blocks_per_stage)
        self.bottleneck = nn.Sequential(*[ResidualBlock(f * 8) for _ in range(res_blocks_per_stage)])

        self.up1 = Up(f * 8, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)
        self.up1.set_res_blocks(concat_ch=(f * 8 + f * 8), out_ch=f * 8, n_blocks=res_blocks_per_stage)
        self.up2.set_res_blocks(concat_ch=(f * 4 + f * 4), out_ch=f * 4, n_blocks=res_blocks_per_stage)
        self.up3.set_res_blocks(concat_ch=(f * 2 + f * 2), out_ch=f * 2, n_blocks=res_blocks_per_stage)
        self.up4.set_res_blocks(concat_ch=(f + f), out_ch=f, n_blocks=res_blocks_per_stage)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

        self.tanh = nn.Tanh()

    def forward(self, x, i_nl=None):
        if i_nl is not None:
            i_rho = x
            inp = torch.cat([i_rho, i_nl], dim=1)
        else:
            inp = x

        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.bottleneck(x5)

        d1 = self.up1(b, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # ✅ Fix: residual scaling prevents black collapse
        residual_raw = self.final_conv(d4)
        residual = 0.1 * self.tanh(residual_raw)  # small correction
        if inp.size(1) == 6:
            i_rho_in = inp[:, :3, :, :]
            derained = torch.clamp(i_rho_in - residual, 0.0, 1.0)
        else:
            derained = None

        return residual, derained
