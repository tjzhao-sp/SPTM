# wespeaker/models/ecapa2_lite.py

from typing import List, Tuple

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise NotImplementedError(
        "activation {} is not implemented".format(name)
    )


def conv2d_out_size(
    size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1
) -> int:
    return (
        size + 2 * padding
        - dilation * (kernel_size - 1)
        - 1
    ) // stride + 1


# =========================================================
# 1. Basic 2D Conv Block
# =========================================================

class Conv2dBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Tuple[int, int] = (1, 1),
        activation: str = "relu",
        bias: bool = False
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias
        )

        self.activation = get_activation(activation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)

        return x


# =========================================================
# 2. Frequency-wise SE
# =========================================================

class FrequencyWiseSqueezeExcitationBlock(nn.Module):

    def __init__(
        self,
        frequency_bins_num: int,
        hidden_dim: int = 64,
        activation: str = "relu"
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(
            frequency_bins_num,
            hidden_dim,
            kernel_size=1,
            bias=True
        )

        self.activation = get_activation(activation)

        self.conv2 = nn.Conv2d(
            hidden_dim,
            frequency_bins_num,
            kernel_size=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, F, T]

        # [B, C, F, T]
        # ->
        # [B, F, C, T]
        y = x.permute(0, 2, 1, 3)

        w = self.pool(y)

        w = self.conv1(w)
        w = self.activation(w)
        w = self.conv2(w)
        w = self.sigmoid(w)

        y = y * w

        # back to [B, C, F, T]
        y = y.permute(0, 2, 1, 3)

        return y


# =========================================================
# 3. Local Feature Extractor Block
# =========================================================

class LocalFeatureExtractorBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_bins_num: int,
        kernel_size: int = 3,
        first_stride: Tuple[int, int] = (1, 1),
        fwse_hidden_dim: int = 64,
        activation: str = "relu",
        use_frequency_encoding: bool = True
    ):
        super().__init__()

        self.conv1 = Conv2dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_stride,
            activation=activation
        )

        self.conv2 = Conv2dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation
        )

        self.conv3 = Conv2dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation
        )

        self.out_frequency_bins_num = conv2d_out_size(
            size=frequency_bins_num,
            kernel_size=kernel_size,
            stride=first_stride[0],
            padding=kernel_size // 2
        )

        self.fwse = FrequencyWiseSqueezeExcitationBlock(
            frequency_bins_num=self.out_frequency_bins_num,
            hidden_dim=fwse_hidden_dim,
            activation=activation
        )

        # =================================================
        # residual projection
        #
        # 原始代码只判断 stride，会导致
        # 1 -> 164 channel 时依赖 broadcasting
        #
        # 这里修复：
        # stride变化 OR channel变化
        # 都进行1x1 projection
        # =================================================

        if (
            first_stride != (1, 1)
            or in_channels != out_channels
        ):

            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.downsample = nn.Identity()

        # Frequency positional encoding
        if use_frequency_encoding:

            self.freq_encoding = nn.Parameter(
                torch.randn(
                    1,
                    1,
                    frequency_bins_num,
                    1
                ) * (frequency_bins_num ** -0.5)
            )

        else:

            self.register_parameter(
                "freq_encoding",
                None
            )

    def forward(self, x):

        residual = self.downsample(x)

        if self.freq_encoding is not None:
            x = x + self.freq_encoding

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.fwse(x)

        x = x + residual

        return x


# =========================================================
# 4. Local Feature Extractor
# =========================================================

class LocalFeatureExtractorModule(nn.Module):

    def __init__(
        self,
        frequency_bins_num: int,
        activation: str = "relu",

        fwse_hidden_dim: int = 64,

        use_frequency_encoding: bool = True,

        out_channels_list: List[int] = [
            64,
            64,
            80,
            96,
            112
        ],

        stride_list: List[Tuple[int, int]] = [
            (1, 1),
            (2, 1),
            (2, 1),
            (2, 1),
            (2, 1)
        ],

        repeat_num_list: List[int] = [
            3,
            4,
            4,
            4,
            5
        ]
    ):

        super().__init__()

        blocks = []

        in_channels = 1

        current_frequency_bins = frequency_bins_num

        for (
            out_channels,
            stage_stride,
            repeat_num
        ) in zip(
            out_channels_list,
            stride_list,
            repeat_num_list
        ):

            for i in range(repeat_num):

                if i == 0:
                    stride = stage_stride
                else:
                    stride = (1, 1)

                block = LocalFeatureExtractorBlock(

                    in_channels=in_channels,

                    out_channels=out_channels,

                    frequency_bins_num=current_frequency_bins,

                    kernel_size=3,

                    first_stride=stride,

                    fwse_hidden_dim=fwse_hidden_dim,

                    activation=activation,

                    use_frequency_encoding=use_frequency_encoding
                )

                blocks.append(block)

                in_channels = out_channels

                current_frequency_bins = (
                    block.out_frequency_bins_num
                )

        self.blocks = nn.ModuleList(blocks)

        self.last_out_channels = in_channels

        self.last_frequency_bins_num = (
            current_frequency_bins
        )

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x


# =========================================================
# 5. Basic 1D Conv Block
# =========================================================

class Conv1dBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: str = "relu",
        bias: bool = False
    ):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias
        )

        self.activation = get_activation(
            activation
        )

        self.bn = nn.BatchNorm1d(
            out_channels
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.activation(x)

        x = self.bn(x)

        return x


# =========================================================
# 6. Res2ConvNet
# =========================================================

class Res2ConvNetModule(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        split_num: int = 4
    ):

        super().__init__()

        assert channels % split_num == 0

        self.width = channels // split_num

        self.split_num = split_num

        self.conv_list = nn.ModuleList([

            nn.Sequential(

                nn.Conv1d(
                    in_channels=self.width,
                    out_channels=self.width,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                ),

                nn.BatchNorm1d(
                    self.width
                )

            )

            for _ in range(
                split_num - 1
            )

        ])

        self.activation = get_activation(
            activation
        )

        self.bn = nn.BatchNorm1d(
            channels
        )

    def forward(self, x):

        # [B,C,T]
        spx = torch.split(
            x,
            self.width,
            dim=1
        )

        # TorchScript friendly
        outputs = torch.jit.annotate(
            List[torch.Tensor],
            []
        )

        sp = spx[0]

        for i, conv in enumerate(
            self.conv_list
        ):

            if i > 0:
                sp = sp + spx[i]

            sp = conv(sp)

            sp = self.activation(sp)

            outputs.append(sp)

        # last split directly bypass
        outputs.append(
            spx[-1]
        )

        out = torch.cat(
            outputs,
            dim=1
        )

        # =================================================
        # 这里修复原始代码的关键BUG：
        #
        # 原版算了 spo，
        # 结果最后 return 的却还是 x
        #
        # 这里真正返回 Res2Conv 的输出
        # =================================================

        out = self.activation(out)

        out = self.bn(out)

        return out


# =========================================================
# 7. 1D SE Block
# =========================================================

class Conv1dSEBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        activation: str = "relu"
    ):

        super().__init__()

        self.pre_conv = Conv1dBlock(

            in_channels=channels,

            out_channels=channels,

            kernel_size=1,

            activation=activation
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

        self.activation = get_activation(
            activation
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.pre_conv(x)

        w = self.pool(x)

        w = self.conv1(w)

        w = self.activation(w)

        w = self.conv2(w)

        w = self.sigmoid(w)

        x = x * w

        return x


# =========================================================
# 8. Global Feature Extractor
# =========================================================

class GlobalFeatureExtractorModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        frequency_bins_num: int,

        hidden_channels: int = 640,

        out_channels: int = 960,

        activation: str = "relu",

        res2convnet_split_num: int = 4
    ):

        super().__init__()

        # flatten C x F
        flat_in_channels = (
            in_channels
            * frequency_bins_num
        )

        # first Conv1D
        self.conv1 = Conv1dBlock(

            in_channels=flat_in_channels,

            out_channels=hidden_channels,

            kernel_size=1,

            activation=activation
        )

        # second Conv1D
        self.conv2 = Conv1dBlock(

            in_channels=hidden_channels,

            out_channels=hidden_channels,

            kernel_size=1,

            activation=activation
        )

        # Res2Conv
        self.res2net = Res2ConvNetModule(

            channels=hidden_channels,

            kernel_size=3,

            activation=activation,

            split_num=res2convnet_split_num
        )

        # SE
        self.se = Conv1dSEBlock(

            channels=hidden_channels,

            activation=activation
        )

        # output projection
        self.post_conv = nn.Conv1d(

            in_channels=hidden_channels,

            out_channels=out_channels,

            kernel_size=1,

            bias=True
        )

        self.activation = get_activation(
            activation
        )

    def forward(self, x):

        # ================================================
        # Input:
        # [B,C,F,T]
        #
        # ->
        #
        # [B,F,C,T]
        # ->
        #
        # [B,F*C,T]
        # ================================================

        x = x.permute(
            0,
            2,
            1,
            3
        ).contiguous()

        x = torch.flatten(
            x,
            start_dim=1,
            end_dim=2
        )

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.res2net(x)

        x = self.se(x)

        x = self.post_conv(x)

        x = self.activation(x)

        return x


# =========================================================
# 9. Channel-dependent statistics pooling
# =========================================================

class ChannelDependentStatisticsPooling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128
    ):

        super().__init__()

        self.in_channels = in_channels

        self.attention = nn.Sequential(

            nn.Conv1d(
                in_channels * 3,
                hidden_channels,
                kernel_size=1,
                bias=True
            ),

            nn.Tanh(),

            nn.Conv1d(
                hidden_channels,
                in_channels,
                kernel_size=1,
                bias=True
            )
        )

    def forward(self, x):

        # x: [B,C,T]

        t = x.size(2)

        mean = torch.mean(
            x,
            dim=2,
            keepdim=True
        )

        std = torch.sqrt(

            torch.var(
                x,
                dim=2,
                keepdim=True,
                unbiased=False
            ).clamp(
                min=1e-5
            )

        )

        # context:
        # x + global mean + global std
        context = torch.cat(

            [
                x,

                mean.expand(
                    -1,
                    -1,
                    t
                ),

                std.expand(
                    -1,
                    -1,
                    t
                )
            ],

            dim=1
        )

        alpha = self.attention(
            context
        )

        alpha = torch.softmax(
            alpha,
            dim=2
        )

        # weighted mean
        mu = torch.sum(
            x * alpha,
            dim=2
        )

        # weighted std
        second_moment = torch.sum(
            (x ** 2) * alpha,
            dim=2
        )

        sg = torch.sqrt(

            (
                second_moment
                - mu ** 2
            ).clamp(
                min=1e-5
            )

        )

        # [B,2C]
        out = torch.cat(
            [
                mu,
                sg
            ],
            dim=1
        )

        return out

    def get_out_dim(self):

        return (
            self.in_channels
            * 2
        )


# =========================================================
# 10. ECAPA2 Lite
# =========================================================

class ECAPA2(nn.Module):

    def __init__(
        self,

        feat_dim: int = 80,

        embed_dim: int = 192,

        emb_bn: bool = False,

        activation: str = "relu",

        lfe_fwse_hidden_dim: int = 64,

        lfe_use_frequency_encoding: bool = True,

        gfe_hidden_channels: int = 640,

        gfe_out_channels: int = 960,

        pool_hidden_channels: int = 128
    ):

        super().__init__()

        # =================================================
        # Local Feature Extractor
        #
        # Original:
        # [164,164,164,192,192]
        #
        # Lite:
        # [64,64,80,96,112]
        #
        # Depth remains:
        # [3,4,4,4,5]
        # =================================================

        self.lfe = LocalFeatureExtractorModule(

            frequency_bins_num=feat_dim,

            activation=activation,

            fwse_hidden_dim=lfe_fwse_hidden_dim,

            use_frequency_encoding=(
                lfe_use_frequency_encoding
            ),

            out_channels_list=[
                64,
                64,
                80,
                96,
                112
            ],

            stride_list=[
                (1, 1),
                (2, 1),
                (2, 1),
                (2, 1),
                (2, 1)
            ],

            repeat_num_list=[
                3,
                4,
                4,
                4,
                5
            ]
        )

        # =================================================
        # Global Feature Extractor
        #
        # Original:
        # hidden = 1024
        # output = 1536
        #
        # Lite:
        # hidden = 640
        # output = 960
        # =================================================

        self.gfe = GlobalFeatureExtractorModule(

            in_channels=(
                self.lfe.last_out_channels
            ),

            frequency_bins_num=(
                self.lfe.last_frequency_bins_num
            ),

            hidden_channels=(
                gfe_hidden_channels
            ),

            out_channels=(
                gfe_out_channels
            ),

            activation=activation,

            res2convnet_split_num=4
        )

        # pooling
        self.pool = (
            ChannelDependentStatisticsPooling(

                in_channels=(
                    gfe_out_channels
                ),

                hidden_channels=(
                    pool_hidden_channels
                )
            )
        )

        # final speaker embedding
        self.embedding = nn.Linear(

            self.pool.get_out_dim(),

            embed_dim
        )

        self.emb_bn = emb_bn

        if emb_bn:

            self.embedding_bn = (
                nn.BatchNorm1d(
                    embed_dim
                )
            )

        else:

            self.embedding_bn = (
                nn.Identity()
            )

    def _get_frame_level_feat(
        self,
        x
    ):

        # =================================================
        # WeSpeaker input:
        #
        # [B,T,F]
        #
        # ECAPA2 LFE requires:
        #
        # [B,1,F,T]
        # =================================================

        x = x.transpose(
            1,
            2
        )

        x = x.unsqueeze(1)

        # Local extractor
        x = self.lfe(x)

        # Global extractor
        x = self.gfe(x)

        # x:
        # [B,960,T]

        return x

    def get_frame_level_feat(
        self,
        x
    ):

        x = self._get_frame_level_feat(
            x
        )

        # WeSpeaker convention
        # [B,T,C]
        return x.transpose(
            1,
            2
        )

    def forward(
        self,
        x
    ):

        # frame-level feature
        frame_feat = (
            self._get_frame_level_feat(
                x
            )
        )

        # attentive statistics pooling
        pooled = self.pool(
            frame_feat
        )

        # speaker embedding
        embedding = self.embedding(
            pooled
        )

        embedding = self.embedding_bn(
            embedding
        )

        # WeSpeaker style
        #
        # outputs[-1]
        # is utterance-level embedding
        return (
            frame_feat,
            embedding
        )


# =========================================================
# Factory
# =========================================================

def E_CAPA2_F80(
    feat_dim: int = 80,
    embed_dim: int = 192,
    emb_bn: bool = False,
    **kwargs
):

    return ECAPA2(

        feat_dim=feat_dim,

        embed_dim=embed_dim,

        emb_bn=emb_bn,

        activation=kwargs.get(
            "activation",
            "relu"
        ),

        lfe_fwse_hidden_dim=kwargs.get(
            "lfe_fwse_hidden_dim",
            64
        ),

        lfe_use_frequency_encoding=kwargs.get(
            "lfe_use_frequency_encoding",
            True
        ),

        gfe_hidden_channels=kwargs.get(
            "gfe_hidden_channels",
            640
        ),

        gfe_out_channels=kwargs.get(
            "gfe_out_channels",
            960
        ),

        pool_hidden_channels=kwargs.get(
            "pool_hidden_channels",
            128
        )
    )


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    model = ECAPA2(
        feat_dim=80,
        embed_dim=192
    )

    model.eval()

    # WeSpeaker feature:
    # B,T,F
    x = torch.randn(
        2,
        200,
        80
    )

    with torch.no_grad():

        frame_feat, emb = model(x)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "frame feature:",
        frame_feat.shape
    )

    print(
        "embedding:",
        emb.shape
    )

    print(
        "parameters: {:.3f} M".format(
            num_params / 1e6
        )
    )

    # Optional TorchScript test
    scripted_model = torch.jit.script(
        model
    )

    print(
        "TorchScript: PASS"
    )

