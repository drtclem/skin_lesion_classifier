import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_model(num_classes=8, device='cpu'):
    """
    Original ResNet50 baseline — unchanged from v1.
    Kept here so you can still train/evaluate it for comparison.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model.to(device)


class HybridSkinLesionModel(nn.Module):
    """
    Hybrid model: ResNet50 + EfficientNetB4 + DenseNet121.

    Each backbone independently extracts a feature vector from the same
    input image. Those three vectors are concatenated and passed through
    a shared classification head.

    Why these three?
      - ResNet50:      deep hierarchical features via skip connections (same as baseline)
      - EfficientNetB4: strong on fine-grained patterns; popular in ISIC competitions
      - DenseNet121:   dense feature reuse; excellent at capturing texture detail,
                       which is clinically relevant for lesion differentiation

    Feature dimensions after global average pooling:
      ResNet50      ->  2048
      EfficientNetB4 -> 1792
      DenseNet121   ->  1024
      ─────────────────────
      Concatenated  ->  4864
    """

    def __init__(self, num_classes=8):
        super().__init__()

        # ── 1. ResNet50 backbone ───────────────────────────────────────────
        # Same freeze/unfreeze strategy as the original model:
        # everything frozen except the last residual block (layer4).
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        # Remove the original FC head. The remaining Sequential ends with
        # AdaptiveAvgPool2d, giving output shape (B, 2048, 1, 1).
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # ── 2. EfficientNetB4 backbone ─────────────────────────────────────
        # EfficientNet's feature extractor is stored in `.features`.
        # We unfreeze only the last conv block (features[-1]) — the same
        # "unfreeze the last block" strategy used on ResNet above.
        effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        for param in effnet.parameters():
            param.requires_grad = False
        for param in effnet.features[-1].parameters():
            param.requires_grad = True
        self.effnet_features = effnet.features          # output: (B, 1792, 7, 7)
        self.effnet_pool = nn.AdaptiveAvgPool2d(1)      # -> (B, 1792, 1, 1)

        # ── 3. DenseNet121 backbone ────────────────────────────────────────
        # DenseNet's layers live in `.features`. We unfreeze denseblock4,
        # the final dense block, which is roughly equivalent to layer4 on ResNet.
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in densenet.parameters():
            param.requires_grad = False
        for param in densenet.features.denseblock4.parameters():
            param.requires_grad = True
        self.densenet_features = densenet.features      # output: (B, 1024, 7, 7)
        self.densenet_pool = nn.AdaptiveAvgPool2d(1)    # -> (B, 1024, 1, 1)

        # ── 4. Fusion classifier ───────────────────────────────────────────
        # Takes the 4864-dim concatenated vector and maps it to class scores.
        # Slightly higher dropout (0.4) on the first layer since the input
        # is much wider than the original 2048.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(4864, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet50: built-in avg pool outputs (B, 2048, 1, 1); flatten -> (B, 2048)
        r = self.resnet(x).flatten(1)

        # EfficientNetB4: features -> pool -> flatten -> (B, 1792)
        e = self.effnet_pool(self.effnet_features(x)).flatten(1)

        # DenseNet121: ReLU is applied manually (DenseNet convention — the
        # final batch norm output needs activation before pooling)
        d = F.relu(self.densenet_features(x))
        d = self.densenet_pool(d).flatten(1)            # (B, 1024)

        # Concatenate all three feature vectors, then classify
        combined = torch.cat([r, e, d], dim=1)          # (B, 4864)
        return self.classifier(combined)


def get_hybrid_model(num_classes=8, device='cpu'):
    return HybridSkinLesionModel(num_classes=num_classes).to(device)