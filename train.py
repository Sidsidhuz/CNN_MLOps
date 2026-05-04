from __future__ import annotations

import argparse
from src.pipeline.train_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the banana leaf disease classifier")
    parser.add_argument("--data-dir", required=True, help="Path to the ImageFolder dataset root")
    parser.add_argument(
        "--output-model-path",
        default=None,
        help="Optional explicit model path. If omitted, saves as artifacts/<crop>_model.pth",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    print("[main] starting training entry point", flush=True)
    args = parse_args()
    print(f"[main] parsed arguments for data_dir={args.data_dir}", flush=True)
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        output_model_path=args.output_model_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    print("[main] invoking TrainingPipeline.run()", flush=True)
    pipeline.run()
    print("[main] training finished", flush=True)


if __name__ == "__main__":
    main()
