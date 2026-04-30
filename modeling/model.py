import torch
import torch.nn as nn
from torchvision import models
from modeling.config import config


class FERModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = config.num_classes
        dropout = config.dropout

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = FERModel()

    dummy_batch = torch.randn(4, 3, 224, 224)
    output = model(dummy_batch)

    print(f"Input shape  : {dummy_batch.shape}")
    print(f"Output shape : {output.shape}")