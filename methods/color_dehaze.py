"""Color-Constrained Dehazing Model.

Based on: Zhang et al. - "Color-Constrained Dehazing Model" (CVPRW 2020)

Key ideas:
- Uses local atmospheric light instead of global
- Adds color constraints based on haze-free image color distributions
- Encoder-decoder with skip connections for direct image restoration
- Color consistency loss to preserve natural colors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class LocalAtmosphereEstimator(nn.Module):
    """Estimates spatially-varying (local) atmospheric light."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ColorConstrainedDehaze(nn.Module):
    """Color-Constrained Dehazing Network.

    Encoder-decoder with:
    - Skip connections for detail preservation
    - Local atmospheric light estimation branch
    - Color-aware restoration
    """

    def __init__(self):
        super().__init__()
        # Main encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
        )

        # Bottleneck with color-aware features
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
        )

        # Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )

        # Local atmosphere estimator
        self.atm_estimator = LocalAtmosphereEstimator()

        # Final restoration: combine features + atmospheric info
        self.final = nn.Sequential(
            nn.Conv2d(64 + 3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) hazy input
        Returns:
            dehazed: (B, 3, H, W) color-constrained dehazed image
            atm_light: (B, 3, H, W) estimated local atmospheric light
        """
        # Local atmospheric light estimation
        atm_light = self.atm_estimator(x)

        # Encoder
        e1 = self.enc1(x)     # 64, H, W
        e2 = self.enc2(e1)    # 128, H/2, W/2
        e3 = self.enc3(e2)    # 256, H/4, W/4
        e4 = self.enc4(e3)    # 512, H/8, W/8

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))

        # Combine with atmospheric light for color-constrained output
        combined = torch.cat([d2, atm_light], dim=1)
        dehazed = self.final(combined)

        # Residual learning: output = input + learned residual
        dehazed = torch.clamp(x + dehazed, 0, 1)

        return dehazed, atm_light


class ColorConsistencyLoss(nn.Module):
    """Color consistency loss to enforce natural color distribution.

    Penalizes deviations from expected color statistics of haze-free images.
    Combines per-channel mean/std matching with inter-channel correlation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """Compute color consistency loss.

        Args:
            output: (B, 3, H, W) predicted dehazed image
            target: (B, 3, H, W) ground truth clean image
        Returns:
            scalar loss
        """
        # Per-channel mean and std matching
        out_mean = output.mean(dim=[2, 3])
        tgt_mean = target.mean(dim=[2, 3])
        out_std = output.std(dim=[2, 3])
        tgt_std = target.std(dim=[2, 3])

        mean_loss = F.mse_loss(out_mean, tgt_mean)
        std_loss = F.mse_loss(out_std, tgt_std)

        # Inter-channel correlation
        B = output.shape[0]
        out_flat = output.view(B, 3, -1)
        tgt_flat = target.view(B, 3, -1)

        out_corr = torch.bmm(out_flat, out_flat.transpose(1, 2)) / out_flat.shape[2]
        tgt_corr = torch.bmm(tgt_flat, tgt_flat.transpose(1, 2)) / tgt_flat.shape[2]

        corr_loss = F.mse_loss(out_corr, tgt_corr)

        return mean_loss + std_loss + 0.5 * corr_loss
