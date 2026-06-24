"""ZenML step for training the CIFAR-10 vision model (ResNet18 transfer learning)."""

import mlflow
import numpy as np
import torch
from components.common import ArrayImageDataset, build_transforms, experiment_tracker_name
from mlflow.models.signature import infer_signature
from components.model import build_model
from torch import nn, optim
from torch.utils.data import DataLoader
from components.tracking import (
    generate_reproduce_run_command,
    get_logbook,
    log_git_info,
    log_model_architecture,
)
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker_name(),
)  # type: ignore[untyped-decorator]
def train_step(  # noqa: PLR0913, PLR0915
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Fine-tune ResNet18 on CIFAR-10 and log artefacts to MLflow.

    Args:
        train_features: Raw uint8 (N, 32, 32, 3) training images.
        train_labels: Training labels array.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        seed: Random seed for reproducibility.

    Returns:
        tuple[torch.nn.Module, dict[str, float]]: Trained model and training metrics.
    """

    mlflow.pytorch.autolog(disable=True)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Held-out validation split, carved out of the raw arrays *before*
    # wrapping in datasets, so train/val each get the right transform
    # (augmentation only on the train side).
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_labels))
    val_size = max(1, int(0.1 * len(indices)))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_dataset = ArrayImageDataset(
        train_features[train_idx], train_labels[train_idx], build_transforms(train=True)
    )
    val_dataset = ArrayImageDataset(
        train_features[val_idx], train_labels[val_idx], build_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    num_classes = int(np.unique(train_labels).size)
    model = build_model(num_classes=num_classes, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_model_architecture(model)
    log_git_info()
    logbook_text = get_logbook()
    mlflow.log_text(logbook_text, "logbook.md")
    mlflow.set_tag("mlflow.note.content", logbook_text)

    active_run = mlflow.active_run()
    if active_run:
        mlflow.log_param(
            "reproduce_command",
            generate_reproduce_run_command(active_run.info.run_id, active_run.info.experiment_id),
        )

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("num_classes", num_classes)

    train_loss_value = 0.0
    val_loss_value = 0.0
    val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss_value = running_loss / len(train_dataset)
        train_accuracy = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss_value = val_loss / len(val_dataset)
        val_accuracy = val_correct / val_total

        mlflow.log_metric("train_loss", train_loss_value, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss_value, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        logger.info(
            "Epoch %s/%s Train Loss: %.4f Train Acc: %.4f Val Loss: %.4f Val Acc: %.4f",
            epoch + 1,
            num_epochs,
            train_loss_value,
            train_accuracy,
            val_loss_value,
            val_accuracy,
        )

    example_input, _ = next(iter(train_loader))
    example_input = example_input[0].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        example_output = model(example_input)

    signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().numpy())

    model_cpu = model.cpu()
    mlflow.pytorch.log_model(
        pytorch_model=model_cpu,
        name="model",
        signature=signature,
        serialization_format="pickle",  # avoids pt2/torch.export
    )
    model.to(device)

    return model, {
        "train_loss": train_loss_value,
        "val_loss": val_loss_value,
        "val_accuracy": val_accuracy,
    }
