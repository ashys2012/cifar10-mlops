"""Shared helpers used by both train_step.py and evaluate_step.py.

Pulled out into one module so the resize/normalize logic and the
experiment-tracker lookup aren't duplicated across the two steps.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from zenml.client import Client

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def experiment_tracker_name() -> str | None:
    """Return the active experiment tracker name if configured.

    Returns:
        str | None: Experiment tracker name if available.
    """
    try:
        tracker = Client().active_stack.experiment_tracker
        return tracker.name if tracker else None
    except Exception:
        return None


class ArrayImageDataset(Dataset):
    """Takes in the numpay arrays and transforms it into tensors for processing
    Args:
        features: A NumPy array containing the raw image data.
        labels: A NumPy array containing the corresponding class labels.
        transform: A torchvision.transforms composition to apply to each image.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, transform: transforms.Compose):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.features[idx])
        return self.transform(img), int(self.labels[idx])


def build_transforms(train: bool) -> transforms.Compose:
    """Resize to what ResNet18 expects; add augmentation only for training."""
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)
