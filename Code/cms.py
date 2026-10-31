import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceAdaptiveEpsSoftmaxCE(nn.Module):
    """
    Confidence-Adaptive epsilon-Softmax Cross Entropy.

    Standard epsilon-softmax:
        p = softmax(logits)
        t = argmax(p)
        p_t = p_t + m
        p = p / (1 + m)

    Proposed confidence-adaptive epsilon:
        m_i = m * p_max^gamma

    Low-confidence samples:
        m_i -> 0, approximately standard AAM + CE.

    High-confidence samples:
        m_i -> m, gradually activating epsilon-softmax.

    Args:
        m: maximum epsilon-softmax modulation strength.
        gamma: confidence modulation exponent.
        eps: numerical stability.
        reduction: mean | sum | none.
    """

    def __init__(self, m=0.05, gamma=2.0, eps=1e-8, reduction="mean"):
        super().__init__()
        if m < 0: raise ValueError(f"m must be >= 0, got {m}")
        if gamma < 0: raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"): raise ValueError(f"reduction must be mean, sum or none, got {reduction}")
        self.m = float(m)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits, target):
        p = F.softmax(logits, dim=1)

        with torch.no_grad():
            p_max, top1 = p.max(dim=1)
            m_i = self.m * p_max.pow(self.gamma)

        p_eps = p.clone()
        batch_idx = torch.arange(p.size(0), device=p.device)
        p_eps[batch_idx, top1] += m_i
        p_eps = p_eps / (1.0 + m_i.unsqueeze(1))

        logp = torch.log(p_eps.clamp_min(self.eps))
        loss = F.nll_loss(logp, target, reduction=self.reduction)

        return loss
