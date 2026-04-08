"""Densely Connected Pyramid Dehazing Network (DCPDN).

Based on: Zhang & Patel - "Densely Connected Pyramid Dehazing Network" (CVPR 2018)

Architecture:
- Pyramid densely-connected encoder-decoder for transmission map estimation
- U-Net for atmospheric light estimation
- Joint discriminator (GAN-based training)
- Atmospheric scattering model layer: J(z) = (I(z) - A(z)(1-t(z))) / t(z)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Dense block with batch norm and ReLU."""

    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate,
                          kernel_size=3, padding=1),
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class PyramidPooling(nn.Module):
    """Pyramid pooling module for multi-scale feature aggregation."""

    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 4
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.conv2 = nn.Conv2d(in_channels, mid, 1)
        self.conv3 = nn.Conv2d(in_channels, mid, 1)
        self.conv4 = nn.Conv2d(in_channels, mid, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        f1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.conv3(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=False)
        f4 = F.interpolate(self.conv4(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([x, f1, f2, f3, f4], dim=1)


class TransmissionEstimator(nn.Module):
    """Pyramid densely-connected encoder-decoder for transmission estimation."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dense1 = DenseBlock(32, 16, 4)  # 32 + 4*16 = 96
        self.down1 = nn.Sequential(
            nn.Conv2d(96, 64, 1),
            nn.AvgPool2d(2),
        )

        self.dense2 = DenseBlock(64, 16, 4)  # 64 + 64 = 128
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.AvgPool2d(2),
        )

        self.dense3 = DenseBlock(128, 32, 4)  # 128 + 128 = 256
        self.pyramid = PyramidPooling(256)  # 256 + 4*64 = 512

        # Decoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(160, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.dense1(e1)  # 96 ch
        p1 = self.down1(d1)   # 64 ch, H/2

        d2 = self.dense2(p1)  # 128 ch
        p2 = self.down2(d2)   # 128 ch, H/4

        d3 = self.dense3(p2)  # 256 ch
        pyr = self.pyramid(d3)  # 512 ch

        up2 = self.up2(pyr)   # 128 ch, H/2
        up2 = self.dec2(torch.cat([up2, d2], dim=1))  # 128 ch

        up1 = self.up1(up2)   # 64 ch, H
        up1 = self.dec1(torch.cat([up1, d1], dim=1))  # 64 ch

        t = self.final(up1)
        return t


class AtmosphericLightEstimator(nn.Module):
    """U-Net style network for atmospheric light estimation."""

    def __init__(self):
        super().__init__()
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)

        self.bottleneck = self._block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = self._block(64, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


class Discriminator(nn.Module):
    """Joint discriminator for GAN training.

    Takes concatenated dehazed image and transmission map as input.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, stride=2, padding=1),  # 3 (image) + 1 (transmission)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, img, t_map):
        x = torch.cat([img, t_map], dim=1)
        return self.model(x)


class DCPDN(nn.Module):
    """Full DCPDN model combining transmission, atmospheric light, and ASM."""

    def __init__(self):
        super().__init__()
        self.transmission_net = TransmissionEstimator()
        self.atmosphere_net = AtmosphericLightEstimator()

    def forward(self, hazy):
        """Forward pass.

        Args:
            hazy: (B, 3, H, W) hazy input
        Returns:
            dehazed: (B, 3, H, W) clean image
            t_map: (B, 1, H, W) transmission map
            A: (B, 3, H, W) atmospheric light map
        """
        t_map = self.transmission_net(hazy)
        A = self.atmosphere_net(hazy)

        # Atmospheric scattering model inversion:
        # J(z) = (I(z) - A(z)(1 - t(z))) / t(z)
        t_clamped = torch.clamp(t_map, min=0.1)
        dehazed = (hazy - A * (1 - t_clamped)) / t_clamped
        dehazed = torch.clamp(dehazed, 0, 1)

        return dehazed, t_map, A
