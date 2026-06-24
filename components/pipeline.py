"""The ZenML pipeline that ties load -> train -> evaluate together."""

from components.data_loader_step import data_loader_step
from components.evaluate_step import evaluate_step
from components.train_step import train_step
from zenml import pipeline


@pipeline
def training_pipeline(
    data_dir: str = "./data",
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> None:
    train_features, train_labels, test_features, test_labels = data_loader_step(
        data_dir=data_dir
    )

    model, _train_metrics = train_step(
        train_features=train_features,
        train_labels=train_labels,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    evaluate_step(
        model=model,
        test_features=test_features,
        test_labels=test_labels,
        batch_size=batch_size,
    )
