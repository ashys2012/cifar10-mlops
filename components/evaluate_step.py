"""Evaluation step for the ZenML pipeline using the CIFAR-10 test split."""

import mlflow
import numpy as np
import torch
from components.common import ArrayImageDataset, build_transforms, experiment_tracker_name
from torch.utils.data import DataLoader
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False, experiment_tracker=experiment_tracker_name()) 
def evaluate_step(
    model: torch.nn.Module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate the trained model on the held-out CIFAR-10 test set.

    Args:
        model: Trained model to evaluate.
        test_features: Raw uint8 (N, 32, 32, 3) test images.
        test_labels: Test labels array.
        batch_size: Batch size for evaluation.

    Returns:
        dict[str, float]: Evaluation metrics for the test split.
    """
    model.eval()
    dataset = ArrayImageDataset(test_features, test_labels, build_transforms(train=False))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss = total_loss / len(test_loader.dataset)
    test_accuracy = correct / total if total else 0.0

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)

    logger.info("Test Loss: %.4f Test Acc: %.4f", test_loss, test_accuracy)

    return {"test_loss": test_loss, "test_accuracy": test_accuracy}
