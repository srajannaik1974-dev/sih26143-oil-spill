"""
ml/training/unet.py
===================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Detection
Vanilla U-Net segmentation architecture.

Architecture overview
---------------------
- Encoder: 4 downsampling blocks, each with 2x Conv→BN→ReLU then MaxPool.
- Bottleneck: 2x Conv→BN→ReLU at the deepest level.
- Decoder: 4 upsampling blocks, each with Upsample→concat skip→2x Conv→BN→ReLU.
- Output: 1x1 Conv → raw logit (no sigmoid).

Input  : (B, 1, H, W) — 1-channel SAR float32, normalised to [0, 1]
Output : (B, 1, H, W) — raw logits (apply sigmoid externally for probabilities)

IMPORTANT: The model outputs RAW LOGITS.
- Use BCEWithLogitsLoss during training (applies sigmoid + BCE internally).
- Apply torch.sigmoid() externally for evaluation/inference only.
- This is numerically more stable than applying sigmoid before BCE.

References
----------
Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block: double convolution
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """
    Two successive Conv2d → BatchNorm2d → ReLU blocks.
    This is the standard U-Net building block.
    Using BatchNorm makes training more stable on small datasets.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder block: DoubleConv + MaxPool
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """
    One level of the U-Net encoder.
    Returns the skip-connection feature map (before pooling) and the
    downsampled feature map (after pooling).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)   # full-resolution feature map for skip connection
        down = self.pool(skip) # half-resolution for next encoder block
        return skip, down


# ---------------------------------------------------------------------------
# Decoder block: Upsample + skip-concat + DoubleConv
# ---------------------------------------------------------------------------

class _DecoderBlock(nn.Module):
    """
    One level of the U-Net decoder.
    Uses bilinear upsampling (instead of transposed convolution) for
    simplicity and to avoid checkerboard artefacts.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # After upsampling + concatenating the skip connection,
        # the channel count is in_ch (upsampled) + in_ch//2 (skip).
        # We handle the flexible channel count in forward() with a
        # dynamic DoubleConv — but here we pre-allocate for the expected
        # in_ch + skip_ch = in_ch + in_ch//2 = 3*in_ch//2.
        # To keep it simple we use in_ch as input (the concatenated channels
        # are computed precisely when called from UNet.__init__).
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv     = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatches that can arise from odd input dimensions.
        # Pad x to match skip's spatial size if necessary.
        if x.shape != skip.shape:
            x = F.pad(x, [
                0, skip.shape[3] - x.shape[3],
                0, skip.shape[2] - x.shape[2],
            ])

        x = torch.cat([skip, x], dim=1)  # concatenate along channel axis
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net for binary segmentation of 1-channel Sentinel-1 SAR imagery.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Must be 1 for SAR (VV).
    out_channels : int
        Number of output channels. 1 for binary segmentation.
    base_features : int
        Number of feature maps in the first encoder block.
        Subsequent blocks double this count.
        Default: 64 (standard U-Net). Use 32 to reduce VRAM usage.

    Forward output
    --------------
    Raw logits of shape (B, 1, H, W).
    Apply torch.sigmoid() externally for probabilities / binary masks.
    """

    def __init__(
        self,
        in_channels:   int = 1,
        out_channels:  int = 1,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        f = base_features  # 64
        # Encoder — each block halves spatial size, doubles features
        self.enc1 = _EncoderBlock(in_channels, f)       # skip: (B, 64, H, W)
        self.enc2 = _EncoderBlock(f,           f * 2)   # skip: (B, 128, H/2, W/2)
        self.enc3 = _EncoderBlock(f * 2,       f * 4)   # skip: (B, 256, H/4, W/4)
        self.enc4 = _EncoderBlock(f * 4,       f * 8)   # skip: (B, 512, H/8, W/8)

        # Bottleneck — no pooling, just two convolutions at the deepest level
        self.bottleneck = _DoubleConv(f * 8, f * 16)    # (B, 1024, H/16, W/16)

        # Decoder — each block doubles spatial size, halves features
        # in_ch = upsampled_channels + skip_channels
        self.dec4 = _DecoderBlock(f * 16 + f * 8,  f * 8)   # 1024+512 -> 512
        self.dec3 = _DecoderBlock(f * 8  + f * 4,  f * 4)   # 512+256  -> 256
        self.dec2 = _DecoderBlock(f * 4  + f * 2,  f * 2)   # 256+128  -> 128
        self.dec1 = _DecoderBlock(f * 2  + f,       f)       # 128+64   -> 64

        # Output head — 1x1 convolution maps to logits, no activation
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float32 tensor, channels normalised to [0, 1]

        Returns
        -------
        logits : (B, 1, H, W) raw logits — apply sigmoid for probabilities
        """
        # Encode
        skip1, x = self.enc1(x)  # skip1: (B, 64, H, W),    x: (B, 64,  H/2, W/2)
        skip2, x = self.enc2(x)  # skip2: (B, 128, H/2, W/2), x: (B, 128, H/4, W/4)
        skip3, x = self.enc3(x)  # skip3: (B, 256, H/4, W/4), x: (B, 256, H/8, W/8)
        skip4, x = self.enc4(x)  # skip4: (B, 512, H/8, W/8), x: (B, 512, H/16,W/16)

        # Bottleneck
        x = self.bottleneck(x)   # (B, 1024, H/16, W/16)

        # Decode (each decoder receives the upsampled x AND the matching skip)
        x = self.dec4(x, skip4)  # (B, 512, H/8,  W/8)
        x = self.dec3(x, skip3)  # (B, 256, H/4,  W/4)
        x = self.dec2(x, skip2)  # (B, 128, H/2,  W/2)
        x = self.dec1(x, skip1)  # (B, 64,  H,    W)

        # Output logits — no sigmoid here
        return self.out_conv(x)  # (B, 1, H, W)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly: python unet.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1, base_features=64)
    print(f"U-Net parameters: {model.count_parameters():,}")

    # Simulate a batch of 2 images at 512x512 with 1 SAR channel
    dummy = torch.randn(2, 1, 512, 512)
    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")   # should be (2, 1, 512, 512)
    assert logits.shape == (2, 1, 512, 512), "Shape check failed!"
    print("Shape check passed.")
