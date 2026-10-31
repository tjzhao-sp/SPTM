# Apache 2.0
# ETDNN for WeSpeaker — OLR20 baseline compatible
# edit by tjzhao
import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers


def _ctx_to_kernel_dil(context):
    """
    将基线里的 context 列表（如 [-2,-1,0,1,2], [-2,0,2], [-3,0,3], [-4,0,4]）
    转换为 (kernel_size, dilation)。默认返回 (1,1)。
    仅处理以 0 为中心且对称且均匀间隔的情形。
    """
    if context is None:
        return 1, 1
    ctx = sorted(context)
    if 0 not in ctx:
        return 1, 1
    # 间隔
    diffs = [ctx[i + 1] - ctx[i] for i in range(len(ctx) - 1)]
    if len(set(diffs)) != 1:
        return 1, 1
    step = diffs[0]
    left, right = abs(ctx[0]), abs(ctx[-1])
    if left != right:
        return 1, 1
    dilation = max(1, step)
    kernel = 2 * left + 1
    return kernel, dilation


class TDNNBlock(nn.Module):
    """Relu + BN + Conv1d TDNN（与 xmuspeech 的 ReluBatchNormTdnnLayer 对齐语义）"""
    def __init__(self, in_ch, out_ch, context=None,
                 bn=True, bn_momentum=0.5, bn_affine=False, bn_track=True,
                 nonlinearity='relu', nonlinearity_params={'inplace': True}):
        super().__init__()
        k, d = _ctx_to_kernel_dil(context)
        pad = (k - 1) // 2 * d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d,
                              padding=pad, bias=False)
        self.use_bn = bn
        if bn:
            self.bn = nn.BatchNorm1d(out_ch, momentum=bn_momentum,
                                     affine=bn_affine, track_running_stats=bn_track)
        act = (nonlinearity or 'relu').lower()
        if act == 'relu':
            self.act = nn.ReLU(**(nonlinearity_params or {}))
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=out_ch)
        elif act in ('silu', 'swish'):
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):  # x: (B, C, T)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for 1D (C, T)."""
    def __init__(self, channels, ratio=4):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):  # (B, C, T)
        s = x.mean(dim=2, keepdim=True)            # GAP over time
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ETDNN(nn.Module):
    """
    WeSpeaker 兼容的 E-TDNN：
      输入: x (B, T, F)
      输出: (embed_a 或 0.0, embed_b)  —— 与 WeSpeaker ResNet 行为一致
    结构按 OLR20 基线：
      tdnn1: ctx [-2,-1,0,1,2], ch=512
      tdnn2: ctx [-2,0,2],      ch=512
      tdnn3: ctx [-3,0,3],      ch=512
      （可选）ex_tdnn4: ctx [-4,0,4], ch=512
      其他 ex_tdnn*、tdnn4/5 为 1×1 TDNN（等价 conv1）
      tdnn5 输出 num_nodes（默认 1500），随后统计池化
      段级：可选 tdnn6（512）、tdnn7（512） → 这里用 seg_1/seg_2 线性层对齐 WeSpeaker
    """
    def __init__(self,
                 feat_dim=80,
                 embed_dim=256,
                 pooling_func='TSTP',
                 two_emb_layer=False,
                 # —— 基线可选开关/参数 —— #
                 extend=False,
                 skip_connection=False,
                 SE=False,
                 se_ratio=4,
                 num_nodes=1500):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.two_emb_layer = two_emb_layer
        self.extend = extend
        self.skip_connection = skip_connection
        self.use_se = SE
        self.se_ratio = se_ratio
        self.num_nodes = num_nodes

        in_ch = feat_dim
        ch = 512  # 基线主干通道

        # ----- Frame-level TDNN 堆栈 -----
        self.tdnn1 = TDNNBlock(in_ch, ch, context=[-2, -1, 0, 1, 2])  # ±2
        self.se1 = SEBlock(ch, ratio=se_ratio) if SE else nn.Identity()

        self.ex_tdnn1 = TDNNBlock(ch, ch) if extend else nn.Identity()

        self.tdnn2 = TDNNBlock(ch, ch, context=[-2, 0, 2])            # ±2（dilation=2）
        self.se2 = SEBlock(ch, ratio=se_ratio) if SE else nn.Identity()
        self.ex_tdnn2 = TDNNBlock(ch, ch) if extend else nn.Identity()

        self.tdnn3 = TDNNBlock(ch, ch, context=[-3, 0, 3])            # ±3（dilation=3）
        self.se3 = SEBlock(ch, ratio=se_ratio) if SE else nn.Identity()
        self.ex_tdnn3 = TDNNBlock(ch, ch) if extend else nn.Identity()

        self.ex_tdnn4 = TDNNBlock(ch, ch, context=[-4, 0, 4]) if extend else nn.Identity()
        self.se4 = SEBlock(ch, ratio=se_ratio) if (SE and extend) else nn.Identity()

        self.ex_tdnn5 = TDNNBlock(ch, ch) if extend else nn.Identity()

        self.tdnn4 = TDNNBlock(ch, ch)  # 1×1

        # 段间捷径：从 tdnn1 输出加到 tdnn4 之后（与基线 skip_connection 语义一致）
        # 这里在 forward 里做

        # 对齐基线：tdnn5 将通道涨到 num_nodes，然后做统计池化
        self.tdnn5 = TDNNBlock(ch, num_nodes)

        # ----- 统计池化（对齐 wespeaker.pooling_layers） -----
        # WeSpeaker 的池化输入维度为通道数
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=num_nodes)
        self.pool_out_dim = self.pool.get_out_dim()

        # ----- Segment / Embedding 层（与 WeSpeaker 风格对齐） -----
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    # ========== 内部：帧级 ==========
    def _get_frame_level_feat(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.tdnn1(x)
        if self.skip_connection:
            identity = x
        else:
            identity = None

        x = self.se1(x)
        x = self.ex_tdnn1(x)

        x = self.tdnn2(x)
        x = self.se2(x)
        x = self.ex_tdnn2(x)

        x = self.tdnn3(x)
        x = self.se3(x)
        x = self.ex_tdnn3(x)
        x = self.ex_tdnn4(x)
        x = self.se4(x)
        x = self.ex_tdnn5(x)

        x = self.tdnn4(x)
        if identity is not None:
            x = x + identity

        x = self.tdnn5(x)  # -> num_nodes 通道
        return x  # (B, C=num_nodes, T)

    # ========== WeSpeaker 对外接口：帧级 ==========
    def get_frame_level_feat(self, x):
        y = self._get_frame_level_feat(x)  # (B, C, T)
        y = y.transpose(1, 2)              # (B, T, C)
        return y

    # ========== 前向 ==========
    def forward(self, x):
        fr = self._get_frame_level_feat(x)  # (B, C, T)
        stats = self.pool(fr)               # (B, pool_out_dim)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


# 方便在 config 里直接选用
def ETDNN_OLR20(feat_dim=80, embed_dim=256, pooling_func='TSTP',
                two_emb_layer=False, extend=False, skip_connection=False,
                SE=False, se_ratio=4, num_nodes=1500):
    return ETDNN(
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        extend=extend,
        skip_connection=skip_connection,
        SE=SE,
        se_ratio=se_ratio,
        num_nodes=num_nodes,
    )


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)  # (B,T,F)
    model = ETDNN_OLR20(feat_dim=80, embed_dim=256, two_emb_layer=True,
                        extend=False, skip_connection=True, SE=False,
                        pooling_func='TSTP', num_nodes=1500)
    model.eval()
    a, b = model(x)
    print(a.shape, b.shape)  # torch.Size([2, 256]) torch.Size([2, 256])
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.3f}M')

    from thop import profile
    x_np = torch.randn(1, 200, 80)
    flops, params = profile(model, inputs=(x_np, ))
    print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
