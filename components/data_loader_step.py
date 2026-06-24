"""ZenML step that downloads CIFAR-10 and returns raw image/label arrays."""

import numpy as np
from torchvision import datasets
from zenml import step
from zenml.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

@step(enable_cache=True)
def data_loader_step(
    data_dir: str = str(DEFAULT_DATA_DIR),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download CIFAR-10 and return raw and untransformed train/test set.

    Args:
        data_dir: Data download directory.

    Returns:
        train_features, train_labels, test_features, test_labels
    """
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)

    train_features = train_dataset.data 
    train_labels = np.array(train_dataset.targets, dtype=np.int64)
    test_features = test_dataset.data
    test_labels = np.array(test_dataset.targets, dtype=np.int64)

    logger.info(
        "Loaded %d train / %d test images across %d classes",
        len(train_labels),
        len(test_labels),
        len(train_dataset.classes),
    )

    return train_features, train_labels, test_features, test_labels
