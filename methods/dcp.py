"""Dark Channel Prior for single image haze removal.

Based on: He, Sun, Tang - "Single Image Haze Removal Using Dark Channel Prior"
"""
import cv2
import numpy as np


class DarkChannelPrior:
    """Dark Channel Prior dehazing method."""

    def __init__(self, patch_size=15, omega=0.95, t0=0.1,
                 guided_filter_radius=60, guided_filter_eps=1e-3):
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.r = guided_filter_radius
        self.eps = guided_filter_eps

    def dark_channel(self, img):
        """Compute the dark channel of an image.

        For each pixel, take the minimum across RGB channels,
        then apply a minimum filter over a local patch.

        Args:
            img: (H, W, 3) float image in [0, 1]
        Returns:
            (H, W) dark channel image
        """
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        dark = cv2.erode(min_channel, kernel)
        return dark

    def estimate_atmospheric_light(self, img, dark):
        """Estimate atmospheric light A from the hazy image.

        Pick the top 0.1% brightest pixels in the dark channel,
        then choose the pixel with highest intensity in the input image.

        Args:
            img: (H, W, 3) hazy image
            dark: (H, W) dark channel
        Returns:
            (3,) atmospheric light vector
        """
        h, w = dark.shape
        num_pixels = h * w
        num_top = max(int(num_pixels * 0.001), 1)

        # Flatten and find top pixels
        dark_flat = dark.ravel()
        indices = np.argpartition(dark_flat, -num_top)[-num_top:]

        # Among these, pick the one with highest intensity in the image
        img_flat = img.reshape(-1, 3)
        intensities = np.sum(img_flat[indices], axis=1)
        best = indices[np.argmax(intensities)]

        A = img_flat[best]
        return A

    def estimate_transmission(self, img, A):
        """Estimate the transmission map.

        t(x) = 1 - omega * dark_channel(I(x) / A)

        Args:
            img: (H, W, 3) hazy image
            A: (3,) atmospheric light
        Returns:
            (H, W) raw transmission map
        """
        normalized = img / (A + 1e-10)
        dark = self.dark_channel(normalized)
        transmission = 1.0 - self.omega * dark
        return transmission

    def guided_filter(self, guide, src, radius, eps):
        """Apply guided filter for transmission refinement.

        Args:
            guide: (H, W) or (H, W, 3) guide image
            src: (H, W) source image to filter
            radius: filter radius
            eps: regularization parameter
        Returns:
            (H, W) filtered image
        """
        if guide.ndim == 3:
            guide_gray = cv2.cvtColor(
                (guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            ).astype(np.float64) / 255.0
        else:
            guide_gray = guide.astype(np.float64)

        src = src.astype(np.float64)
        mean_I = cv2.boxFilter(guide_gray, -1, (radius, radius))
        mean_p = cv2.boxFilter(src, -1, (radius, radius))
        corr_I = cv2.boxFilter(guide_gray * guide_gray, -1, (radius, radius))
        corr_Ip = cv2.boxFilter(guide_gray * src, -1, (radius, radius))

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide_gray + mean_b

    def recover(self, img, transmission, A):
        """Recover the haze-free image.

        J(x) = (I(x) - A) / max(t(x), t0) + A

        Args:
            img: (H, W, 3) hazy image
            transmission: (H, W) refined transmission map
            A: (3,) atmospheric light
        Returns:
            (H, W, 3) recovered image clipped to [0, 1]
        """
        t = np.maximum(transmission, self.t0)
        J = (img - A) / t[:, :, np.newaxis] + A
        return np.clip(J, 0, 1)

    def dehaze(self, img):
        """Run the full DCP dehazing pipeline.

        Args:
            img: (H, W, 3) float image in [0, 1] or uint8 [0, 255]
        Returns:
            (H, W, 3) dehazed image in [0, 1]
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0
        else:
            img = img.astype(np.float64)

        # Step 1: Dark channel
        dark = self.dark_channel(img)

        # Step 2: Atmospheric light
        A = self.estimate_atmospheric_light(img, dark)

        # Step 3: Transmission estimation
        transmission = self.estimate_transmission(img, A)

        # Step 4: Guided filter refinement
        transmission_refined = self.guided_filter(
            img, transmission, self.r, self.eps)
        transmission_refined = np.clip(transmission_refined, 0, 1)

        # Step 5: Scene recovery
        result = self.recover(img, transmission_refined, A)
        return result.astype(np.float32)
