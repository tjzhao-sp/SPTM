import math
from typing import Sequence

import torch
import torch.nn as nn

import wespeaker.models.pooling_layers as pooling_layers


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class SKAttention1D(nn.Module):
    """Multi-scale Selective Kernel Attention."""

    def __init__(self, channels, reduction=4, min_dim=16, num_kernels=2):
        super().__init__()

        hidden = max(min_dim, channels // reduction)

        self.fc = nn.Linear(channels, hidden)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(num_kernels)])

    def forward(self, x, convs):
        # Different convolution kernels
        feats = torch.stack([conv(x) for conv in convs], dim=0)  # [K, B, C, T]

        # Fuse branches
        fused = feats.sum(dim=0)

        # Global descriptor
        descriptor = fused.mean(dim=-1)
        z = self.relu(self.fc(descriptor))

        weights = torch.stack([fc(z).unsqueeze(-1) for fc in self.fcs], dim=0)
        weights = torch.softmax(weights, dim=0)

        out = (weights * feats).sum(dim=0)
        return out


class MSKABottle2neck(nn.Module):
    """
    Multi-scale SKA Bottle2neck.

    Core msSKA module in SKA-TDNN.
    """

    def __init__(self, inplanes, planes, dilation, scale=8, kernel_sizes=(5, 7), group=1):
        super().__init__()

        if planes % scale != 0:
            raise ValueError(f"planes={planes} must be divisible by scale={scale}")

        width = int(math.floor(planes / scale))

        self.width = width
        self.nums = scale - 1

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(width * scale)

        # Each Res2Net branch contains
        # multiple convolution kernels.
        self.skconvs = nn.ModuleList()

        for _ in range(self.nums):
            branches = nn.ModuleList()

            for k in kernel_sizes:
                branches.append(
                    nn.Sequential(
                        nn.Conv1d(
                            width,
                            width,
                            kernel_size=k,
                            dilation=dilation,
                            padding=(k // 2) * dilation,
                            groups=group,
                        ),
                        nn.ReLU(),
                        nn.BatchNorm1d(width),
                    )
                )

            self.skconvs.append(branches)

        self.sk_att = SKAttention1D(
            channels=width,
            reduction=4,
            num_kernels=len(kernel_sizes),
        )

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        splits = torch.split(out, self.width, dim=1)

        outputs = []
        sp = None

        for i in range(self.nums):
            if i == 0:
                sp = splits[i]
            else:
                sp = sp + splits[i]

            sp = self.sk_att(sp, self.skconvs[i])
            outputs.append(sp)

        outputs.append(splits[self.nums])

        out = torch.cat(outputs, dim=1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out = out + residual

        return out


class FrequencySKAttention(nn.Module):
    """
    Frequency-wise SKA.

    Input:
        [B, C, F, T]
    """

    def __init__(
        self,
        freq,
        channels=128,
        kernels=(5, 7),
        dilations=(1, 1),
        reduction=8,
        groups=1,
        min_dim=16,
    ):
        super().__init__()

        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have equal length")

        self.freq = freq
        self.convs = nn.ModuleList()

        for k, d in zip(kernels, dilations):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=k,
                        padding=(k // 2) * d,
                        dilation=d,
                        groups=groups,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(channels),
                )
            )

        hidden = max(min_dim, freq // reduction)

        self.fc = nn.Linear(freq, hidden)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([nn.Linear(hidden, freq) for _ in range(len(kernels))])

    def forward(self, x):
        # x:
        # [B, C, F, T]

        feats = torch.stack([conv(x) for conv in self.convs], dim=0)
        fused = feats.sum(dim=0)

        # Average channel and time,
        # preserve frequency.
        descriptor = fused.mean(dim=(1, 3))

        z = self.relu(self.fc(descriptor))

        weights = torch.stack(
            [fc(z).view(x.size(0), 1, self.freq, 1) for fc in self.fcs],
            dim=0
        )

        weights = torch.softmax(weights, dim=0)
        out = (weights * feats).sum(dim=0)

        return out


class ChannelSKAttention(nn.Module):
    """
    Channel-wise SKA.

    Input:
        [B, C, F, T]
    """

    def __init__(
        self,
        channels=128,
        kernels=(5, 7),
        dilations=(1, 1),
        reduction=8,
        groups=1,
        min_dim=16,
    ):
        super().__init__()

        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have equal length")

        self.channels = channels
        self.convs = nn.ModuleList()

        for k, d in zip(kernels, dilations):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=k,
                        padding=(k // 2) * d,
                        dilation=d,
                        groups=groups,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(channels),
                )
            )

        hidden = max(min_dim, channels // reduction)

        self.fc = nn.Linear(channels, hidden)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(len(kernels))])

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=0)
        fused = feats.sum(dim=0)

        # Average frequency + time,
        # preserve channel.
        descriptor = fused.mean(dim=(2, 3))

        z = self.relu(self.fc(descriptor))

        weights = torch.stack(
            [fc(z).view(x.size(0), self.channels, 1, 1) for fc in self.fcs],
            dim=0
        )

        weights = torch.softmax(weights, dim=0)
        out = (weights * feats).sum(dim=0)

        return out


class FreqChannelSKResBlock(nn.Module):
    """
    Frequency-channel-wise SKA block.

    fcwSKA front-end module.
    """

    def __init__(self, channels, freq, reduction=8):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channels)

        self.freq_att = FrequencySKAttention(
            freq=freq,
            channels=channels,
            kernels=(5, 7),
            dilations=(1, 1),
            reduction=reduction,
        )

        self.channel_att = ChannelSKAttention(
            channels=channels,
            kernels=(5, 7),
            dilations=(1, 1),
            reduction=reduction,
        )

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)

        out = self.freq_att(out)
        out = self.channel_att(out)

        out = out + residual
        out = self.relu(out)

        return out


class SKACCSP(nn.Module):
    """
    Channel- and Context-Dependent
    Statistics Pooling.

    This corresponds to the pooling
    used by the original SKA-TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, **kwargs):
        super().__init__()

        self.in_dim = in_dim

        self.attention = nn.Sequential(
            nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # x:
        # [B, C, T]

        mean = x.mean(dim=2, keepdim=True)
        mean = mean.expand_as(x)

        std = torch.sqrt(
            torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)
        )
        std = std.expand_as(x)

        context = torch.cat((x, mean, std), dim=1)
        weight = self.attention(context)

        mu = torch.sum(x * weight, dim=2)

        sg = torch.sqrt(
            (
                torch.sum((x ** 2) * weight, dim=2)
                - mu ** 2
            ).clamp(min=1e-4)
        )

        stats = torch.cat((mu, sg), dim=1)
        return stats

    def get_out_dim(self):
        return self.in_dim * 2


class SKAASP(nn.Module):
    """
    Standard attentive statistics
    pooling variant.
    """

    def __init__(self, in_dim, bottleneck_dim=128, **kwargs):
        super().__init__()

        self.in_dim = in_dim

        self.attention = nn.Sequential(
            nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True).expand_as(x)

        std = torch.sqrt(
            torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)
        ).expand_as(x)

        context = torch.cat((x, mean, std), dim=1)
        weight = self.attention(context)

        mu = torch.sum(x * weight, dim=2)

        sg = torch.sqrt(
            (
                torch.sum((x ** 2) * weight, dim=2)
                - mu ** 2
            ).clamp(min=1e-4)
        )

        return torch.cat((mu, sg), dim=1)

    def get_out_dim(self):
        return self.in_dim * 2


class SKA_TDNN(nn.Module):

    def __init__(
        self,
        feat_dim=80,
        embed_dim=192,
        channels=1024,
        scale=8,
        pooling_func="SKACCSP",
        global_context_att=True,
        emb_bn=True,
        front_channels=128,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.channels = channels
        self.scale = scale

        # ==================================
        # Frequency dimension calculation
        # ==================================

        # 80 -> 40
        freq_after_conv1 = (feat_dim + 1) // 2

        # 40 -> 20
        freq_after_conv2 = (freq_after_conv1 + 1) // 2

        # ==================================
        # fcwSKA Front-end
        # ==================================

        self.front_conv1 = nn.Conv2d(
            1,
            front_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
        )

        self.front_bn1 = nn.BatchNorm2d(front_channels)

        self.front_block1 = FreqChannelSKResBlock(
            channels=front_channels,
            freq=freq_after_conv1,
        )

        self.front_block2 = FreqChannelSKResBlock(
            channels=front_channels,
            freq=freq_after_conv1,
        )

        self.front_conv2 = nn.Conv2d(
            front_channels,
            front_channels,
            kernel_size=3,
            stride=(2, 2),
            padding=1,
        )

        self.front_bn2 = nn.BatchNorm2d(front_channels)

        # ==================================
        # TDNN
        # ==================================

        self.conv1 = nn.Conv1d(
            front_channels * freq_after_conv2,
            channels,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)

        # ==================================
        # msSKA blocks
        # ==================================

        self.layer1 = MSKABottle2neck(
            channels,
            channels,
            dilation=2,
            scale=scale,
        )

        self.layer2 = MSKABottle2neck(
            channels,
            channels,
            dilation=3,
            scale=scale,
        )

        self.layer3 = MSKABottle2neck(
            channels,
            channels,
            dilation=4,
            scale=scale,
        )

        # ==================================
        # Multi-layer aggregation
        # ==================================

        self.frame_dim = 1536

        self.layer4 = nn.Conv1d(
            3 * channels,
            self.frame_dim,
            kernel_size=1,
        )

        # ==================================
        # Pooling
        # ==================================

        if pooling_func == "SKACCSP":
            self.pool = SKACCSP(in_dim=self.frame_dim)

        elif pooling_func == "SKAASP":
            self.pool = SKAASP(in_dim=self.frame_dim)

        else:
            self.pool = getattr(pooling_layers, pooling_func)(
                in_dim=self.frame_dim,
                global_context_att=global_context_att,
            )

        self.pool_out_dim = self.pool.get_out_dim()

        # ==================================
        # Embedding
        # ==================================

        self.bn5 = nn.BatchNorm1d(self.pool_out_dim)
        self.fc6 = nn.Linear(self.pool_out_dim, embed_dim)

        if emb_bn:
            self.bn6 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn6 = nn.Identity()

    def _get_frame_level_feat(self, x):
        # WeSpeaker input:
        # [B, T, F]

        if x.dim() != 3:
            raise ValueError("SKA_TDNN input must have shape [B, T, F]")

        if x.size(-1) != self.feat_dim:
            raise ValueError(f"Expected feat_dim={self.feat_dim}, got {x.size(-1)}")

        # ==================================
        # [B,T,F]
        # ->
        # [B,1,F,T]
        # ==================================

        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)

        # ==================================
        # fcwSKA front-end
        # ==================================

        x = self.front_conv1(x)
        x = self.relu(x)
        x = self.front_bn1(x)

        x = self.front_block1(x)
        x = self.front_block2(x)

        x = self.front_conv2(x)
        x = self.relu(x)
        x = self.front_bn2(x)

        # x:
        # [B, 128, 20, T/2]

        # ==================================
        # Frequency × Channel flatten
        # ==================================

        x = x.flatten(start_dim=1, end_dim=2)

        # x:
        # [B, 2560, T/2]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # ==================================
        # msSKA TDNN
        # ==================================

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        # ==================================
        # Multi-layer aggregation
        # ==================================

        out = torch.cat((x1, x2, x3), dim=1)
        out = self.layer4(out)
        out = self.relu(out)

        # [B, 1536, T/2]

        return out

    def get_frame_level_feat(self, x):
        x = self._get_frame_level_feat(x)

        # WeSpeaker style:
        # [B, T, D]

        return x.permute(0, 2, 1)

    def forward(self, x):
        # ==================================
        # Frame-level encoder
        # ==================================

        x = self._get_frame_level_feat(x)

        # ==================================
        # Pooling
        # ==================================

        stats = self.pool(x)
        stats = self.bn5(stats)

        # ==================================
        # Embedding
        # ==================================

        embedding = self.fc6(stats)
        embedding = self.bn6(embedding)

        # WeSpeaker executor normally
        # takes outputs[-1] as embedding.

        dummy = torch.tensor(0.0, device=embedding.device)

        return dummy, embedding


def SKA_TDNN_c1024(
    feat_dim=80,
    embed_dim=192,
    pooling_func="SKACCSP",
    global_context_att=True,
    emb_bn=True,
):
    return SKA_TDNN(
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        channels=1024,
        scale=8,
        pooling_func=pooling_func,
        global_context_att=global_context_att,
        emb_bn=emb_bn,
    )


if __name__ == "__main__":
    model = SKA_TDNN_c1024(feat_dim=80, embed_dim=192)

    model.eval()

    x = torch.randn(2, 200, 80)

    with torch.no_grad():
        _, emb = model(x)

    print("input:", x.shape)
    print("embedding:", emb.shape)
    print("params(M):", sum(p.numel() for p in model.parameters()) / 1e6)