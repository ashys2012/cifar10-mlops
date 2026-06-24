"""CLI entrypoint: configure and trigger a run of the training pipeline.

Usage:
    python run.py
    python run.py --epochs 15 --batch-size 128 --lr 0.0005
"""
import argparse

from components.pipeline import training_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on CIFAR-10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_pipeline(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
