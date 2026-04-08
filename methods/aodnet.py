"""AOD-Net: All-in-One Dehazing Network.

Based on: Li et al. - "AOD-Net: All-In-One Dehazing Network" (ICCV 2017)

The key idea is to reformulate the atmospheric scattering model:
    I(x) = J(x)t(x) + A(1 - t(x))

Into a single variable K(x):
    J(x) = K(x) * I(x) - K(x) + b

where K(x) is estimated by a lightweight CNN.
"""
import torch
import torch.nn as nn


class AODNet(nn.Module):
    """AOD-Net architecture.

    Five convolutional layers with feature concatenation.
    Estimates K(x) which is then used to recover the clean image.
    """

    def __init__(self):
        super().__init__()
        # Layer 1: 3 -> 3, kernel 1x1
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(3)

        # Layer 2: 3 -> 3, kernel 3x3
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(3)

        # Layer 3: 6 -> 3, kernel 5x5 (concat of conv1 and conv2 outputs)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(3)

        # Layer 4: 6 -> 3, kernel 7x7 (concat of conv2 and conv3 outputs)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm2d(3)

        # Layer 5: 12 -> 3, kernel 3x3 (concat of conv1,2,3,4)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) hazy input image
        Returns:
            (B, 3, H, W) dehazed image
        """
        # Feature extraction with dense connections
        f1 = self.relu(self.bn1(self.conv1(x)))
        f2 = self.relu(self.bn2(self.conv2(f1)))
        cat1 = torch.cat([f1, f2], dim=1)  # 6 channels

        f3 = self.relu(self.bn3(self.conv3(cat1)))
        cat2 = torch.cat([f2, f3], dim=1)  # 6 channels

        f4 = self.relu(self.bn4(self.conv4(cat2)))
        cat3 = torch.cat([f1, f2, f3, f4], dim=1)  # 12 channels

        # K(x) estimation
        K = self.conv5(cat3)

        # Clean image generation: J(x) = K(x) * I(x) - K(x) + b
        b = 1.0
        output = K * x - K + b

        return torch.clamp(output, 0, 1)
