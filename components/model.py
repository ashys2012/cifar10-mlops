"""Model construction: a pre-trained ResNet18 with the classifier head
replaced for the target number of classes (10, for CIFAR-10).
"""
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)
