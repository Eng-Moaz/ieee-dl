"""
ResNet-50-based model for Facial Emotion Recognition.

The architecture follows a transfer-learning approach: the pre-trained
ResNet-50 backbone extracts rich visual features, while a lightweight
classification head maps those features to the target emotion classes.
"""

import torch
import torch.nn as nn
from torchvision import models
from modeling.config import config


class FERModel(nn.Module):
    """Facial Emotion Recognition model built on a pre-trained ResNet-50 backbone.

    The backbone is frozen by default to support Phase 1 (head-only) training.
    Call :meth:`unfreeze_backbone` before Phase 2 to enable full fine-tuning.

    Architecture:
        - **Backbone**: ResNet-50 (ImageNet pre-trained, global average pooling output → 2048-d)
        - **Head**: Linear(2048, 256) → ReLU → Dropout → Linear(256, num_classes)
    """

    def __init__(self):
        super().__init__()
        num_classes = config.num_classes
        dropout = config.dropout

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in backbone.parameters():
            param.requires_grad = False

        # Strip the original FC layer; retain everything up to global average pooling.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw class logits.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(N, num_classes)``.
        """
        x = self.backbone(x)
        x = self.head(x)
        return x

    def unfreeze_backbone(self) -> None:
        """Enable gradient computation for all backbone parameters.

        Call this method before Phase 2 training to allow the backbone weights
        to be updated during full fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = FERModel()

    dummy_batch = torch.randn(4, 3, 224, 224)
    output = model(dummy_batch)

    print(f"Input shape  : {dummy_batch.shape}")
    print(f"Output shape : {output.shape}")