"""
Focal Loss for multi-class classification under class imbalance.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional per-class weighting.

    Focal Loss down-weights well-classified examples by multiplying the
    standard cross-entropy loss by a modulating factor ``(1 - p_t) ** gamma``.
    This concentrates training on hard, misclassified samples and is
    especially effective when dealing with severe class imbalance.

    Args:
        alpha: Per-class weight tensor of shape ``(num_classes,)``.  When
            provided, each sample's loss is scaled by the weight of its
            true class.  Pass ``None`` to disable class weighting.
        gamma: Focusing exponent.  Higher values suppress easy examples more
            aggressively.  Typical values are in the range ``[0.5, 5]``.
        reduction: Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'`` | ``'none'``.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss between ``inputs`` and ``targets``.

        Args:
            inputs: Raw class logits of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Scalar loss value (or per-sample tensor when ``reduction='none'``).
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
