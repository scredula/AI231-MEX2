"""
From-scratch CNN architectures for keyword spotting (wake word).

All models take a log-mel spectrogram (B, 1, n_mels, T) and return logits
(B, num_classes). They keep the mel axis (2-D convolutions) so frequency
information is preserved — unlike a 1-D model that averages mel away.

Architectures:
  - dscnn          : depthwise-separable CNN ("Hello Edge")
  - matchboxnet2d  : residual depthwise-separable CNN (MatchboxNet/Bc-ResNet)
  - tcresnet       : temporal-conv ResNet, (k,1) kernels spanning all mels
  - vgg_small      : small VGG-style CNN (strong baseline)

Select with config.model.arch.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DepthwiseSeparable2d(nn.Module):
    """depthwise kxk -> pointwise 1x1 (each + BN + ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))


class SqueezeExcite2d(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        h = max(1, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, h, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResSepBlock2d(nn.Module):
    """MatchboxNet-style sub-block: 1x1 -> depthwise kxk -> 1x1, residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 11,
                 dilation: int = 1, dropout: float = 0.0, se: bool = False):
        super().__init__()
        pad = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation,
                      groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.se = SqueezeExcite2d(out_ch) if se else None
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.res = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch))
                    if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.se is not None:
            out = self.se(out)
        out = self.drop(out) + self.res(x)
        return self.relu(out)


# ---------------------------------------------------------------------------
# 1) DS-CNN
# ---------------------------------------------------------------------------
class DSCNN(nn.Module):
    def __init__(self, input_channels=1, n_mels=64, num_classes=1,
                 channels: int = 64, num_blocks: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(DepthwiseSeparable2d(channels, channels, 3))
            blocks.append(nn.MaxPool2d(2))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(channels, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2) TC-ResNet
# ---------------------------------------------------------------------------
class TCResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (k, 1), padding=(k // 2, 0), bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (k, 1), padding=(k // 2, 0), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.res = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch))
                    if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.res(x))


class TCResNet(nn.Module):
    def __init__(self, input_channels=1, n_mels=64, num_classes=1,
                 channels=(16, 24, 32, 48), k: int = 9, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(1, 2), padding=(1, 0)),
        )
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(TCResBlock(channels[i], channels[i + 1], k=k))
            blocks.append(nn.MaxPool2d((1, 2)))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(channels[-1], num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3) MatchboxNet (2-D residual depthwise-separable)
# ---------------------------------------------------------------------------
class MatchboxNet2D(nn.Module):
    def __init__(self, input_channels=1, n_mels=64, num_classes=1,
                 stem_channels: int = 64, stem_kernel: int = 5,
                 channels=(64, 128, 256), kernels=(11, 13, 15),
                 blocks_per_stage=(2, 2, 2), head_channels: int = 128,
                 dropout: float = 0.2, se: bool = False):
        super().__init__()
        self.num_classes = num_classes
        sp = stem_kernel // 2
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, stem_kernel, padding=sp, bias=False),
            nn.BatchNorm2d(stem_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        stages = []
        in_ch = stem_channels
        for c, k, nb in zip(channels, kernels, blocks_per_stage):
            for j in range(nb):
                stages.append(ResSepBlock2d(in_ch, c, kernel_size=k,
                                            dropout=dropout if j == nb - 1 else 0.0, se=se))
                in_ch = c
            stages.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(head_channels, num_classes),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 4) Small VGG-style CNN
# ---------------------------------------------------------------------------
class VGGSmall(nn.Module):
    def __init__(self, input_channels=1, n_mels=64, num_classes=1,
                 channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        conv = []
        in_ch = input_channels
        for c in channels:
            conv += [nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
                     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                     nn.Conv2d(c, c, 3, padding=1, bias=False),
                     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                     nn.MaxPool2d(2)]
            in_ch = c
        self.features = nn.Sequential(*conv)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_ARCHS = {
    "dscnn": DSCNN,
    "tcresnet": TCResNet,
    "matchboxnet2d": MatchboxNet2D,
    "vgg_small": VGGSmall,
}


def create_model(config):
    """Build a model from config.model (arch + optional hyperparameters)."""
    m = config.model
    arch = str(m.get("arch", "matchboxnet2d")).lower()
    if arch not in _ARCHS:
        raise ValueError(f"Unknown model.arch '{arch}'. Options: {list(_ARCHS)}")

    common = dict(input_channels=m.get("input_channels", 1),
                  n_mels=m.get("n_mels", 64),
                  num_classes=m.get("num_classes", 1))

    if arch == "dscnn":
        ch = m.get("dscnn_channels", 64)
        ch = int(ch[0] if isinstance(ch, (list, tuple)) else ch)
        return DSCNN(channels=ch,
                     num_blocks=int(m.get("dscnn_blocks", 4)),
                     dropout=float(m.get("dropout", 0.2)), **common)
    if arch == "tcresnet":
        return TCResNet(channels=tuple(m.get("tcresnet_channels", [16, 24, 32, 48])),
                        k=int(m.get("tcresnet_kernel", 9)),
                        dropout=float(m.get("dropout", 0.2)), **common)
    if arch == "vgg_small":
        return VGGSmall(channels=tuple(m.get("vgg_channels", [32, 64, 128, 256])),
                        dropout=float(m.get("dropout", 0.3)), **common)

    return MatchboxNet2D(
        stem_channels=int(m.get("stem_channels", 64)),
        stem_kernel=int(m.get("stem_kernel", 5)),
        channels=tuple(m.get("channels", [64, 128, 256])),
        kernels=tuple(m.get("kernels", [11, 13, 15])),
        blocks_per_stage=tuple(m.get("blocks_per_stage", [2, 2, 2])),
        head_channels=int(m.get("head_channels", 128)),
        dropout=float(m.get("dropout", 0.2)),
        se=bool(m.get("se", False)),
        **common,
    )

