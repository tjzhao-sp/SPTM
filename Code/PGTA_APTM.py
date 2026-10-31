import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias) for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class PGTA(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128, kernel_size=9, init_beta=0.9, init_alpha=0.7, temperature=1.5, prior_ratio=0.10, eps=1e-6, detach_prior=True, clamp_q=5.0):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

        assert kernel_size >= 3 and kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.temperature = float(temperature)
        self.prior_ratio = float(prior_ratio)
        self.eps = float(eps)
        self.detach_prior = bool(detach_prior)
        self.clamp_q = float(clamp_q)

        init_alpha = min(max(float(init_alpha), 1e-4), 1 - 1e-4)
        init_beta = min(max(float(init_beta), 1e-4), 1 - 1e-4)

        self.alpha_logit = nn.Parameter(torch.log(torch.tensor(init_alpha / (1 - init_alpha))))
        self.beta_logit = nn.Parameter(torch.log(torch.tensor(init_beta / (1 - init_beta))))

        r = kernel_size // 2
        dist = torch.arange(-r, r + 1).abs().float()
        self.register_buffer("dist", dist, persistent=False)

    def _build_decay_kernel(self, device, dtype):
        beta = torch.sigmoid(self.beta_logit).to(device=device, dtype=dtype)
        k = torch.pow(beta, self.dist.to(device=device, dtype=dtype))
        k = k / (k.sum() + self.eps)
        return k.view(1, 1, -1)

    def forward(self, x):
        x_for_prior = x.detach() if self.detach_prior else x

        q = x_for_prior.abs().mean(dim=1)
        q = q - q.mean(dim=1, keepdim=True)
        q = q / (q.std(dim=1, keepdim=True) + self.eps)

        if self.clamp_q > 0:
            q = torch.clamp(q, -self.clamp_q, self.clamp_q)

        a = F.softmax(q / max(self.temperature, self.eps), dim=1)

        k = self._build_decay_kernel(device=x.device, dtype=x.dtype)
        a_smooth = F.conv1d(q.unsqueeze(1), k, padding=self.kernel_size // 2).squeeze(1)

        alpha = torch.sigmoid(self.alpha_logit).to(device=x.device, dtype=x.dtype)
        w = alpha * a + (1.0 - alpha) * a_smooth
        w = torch.clamp(w, min=0.0)
        w = w / (w.sum(dim=1, keepdim=True) + self.eps)

        z_global = x.mean(dim=2)
        z_prior = torch.sum(x * w.unsqueeze(1), dim=2)
        z = (1.0 - self.prior_ratio) * z_global + self.prior_ratio * z_prior

        g = F.relu(self.linear1(z))
        g = torch.sigmoid(self.linear2(g))

        return x * g.unsqueeze(2)


class APTM_Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            PGTA(channels, se_bottleneck_dim=128, kernel_size=9, temperature=1.5, prior_ratio=0.10)
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class SPTM_nest(nn.Module):
    def __init__(self, channels=512, feat_dim=80, embed_dim=192, pooling_func='ASTP', global_context_att=False, emb_bn=False):
        super().__init__()

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer_ = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = APTM_Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = APTM_Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = APTM_Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        self.fusion_scale1 = nn.Parameter(torch.tensor(0.5))
        self.fusion_scale2 = nn.Parameter(torch.tensor(0.5))
        self.fusion_scale3 = nn.Parameter(torch.tensor(0.5))

        cat_channels = channels * 3
        out_channels = 512 * 3

        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=out_channels, global_context_att=global_context_att)
        self.pool_out_dim = self.pool.get_out_dim()

        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)

        self.emb_bn = emb_bn
        self.bn2 = nn.BatchNorm1d(embed_dim) if emb_bn else nn.Identity()

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        lam1 = torch.sigmoid(self.fusion_scale1)
        lam2 = torch.sigmoid(self.fusion_scale2)
        lam3 = torch.sigmoid(self.fusion_scale3)

        out1 = self.layer1(x)
        out_ = self.layer_(x)
        out2 = self.layer2(lam1 * out_ + (1.0 - lam1) * out1)
        out3 = self.layer3(lam2 * out1 + (1.0 - lam2) * out2)
        out4 = self.layer4(lam3 * out1 + (1.0 - lam3) * out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)

        return out, out4

    def get_frame_level_feat(self, x):
        return self._get_frame_level_feat(x)[0].permute(0, 2, 1)

    def forward(self, x):
        out, out4 = self._get_frame_level_feat(x)
        out = F.relu(out)
        out = self.bn(self.pool(out))
        out = self.linear(out)

        if self.emb_bn:
            out = self.bn2(out)

        return out4, out


def SPTM(feat_dim, embed_dim, pooling_func='ASTP', emb_bn=False):
    return SPTM_nest(channels=512, feat_dim=feat_dim, embed_dim=embed_dim, pooling_func=pooling_func, global_context_att=True, emb_bn=emb_bn)


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)

    model = SPTM(feat_dim=80, embed_dim=256, pooling_func='ASTP')
    model.eval()

    out = model(x)
    print(out[-1].shape)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))

    from thop import profile

    x_np = torch.randn(1, 200, 80)
    flops, params = profile(model, inputs=(x_np,))
    print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))