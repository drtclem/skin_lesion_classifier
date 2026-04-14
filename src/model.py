
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=8, device='cpu'):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last ResNet block
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    
    return model.to(device)