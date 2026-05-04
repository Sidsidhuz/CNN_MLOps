from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.components.artifacts import DataSplitArtifact
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.model import LeafDiseaseCNN
from src.components.trainer import ModelTrainer


class TrainingPipelineConfig:
    def __init__(
        self,
        data_dir: str,
        output_model_path: str = "artifacts/model.pth",
        image_size: int = 150,
        batch_size: int = 32,
        val_split: float = 0.2,
        seed: int = 42,
        epochs: int = 25,
        learning_rate: float = 1e-3,
    ) -> None:
        self.data_dir = data_dir
        self.output_model_path = output_model_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate


class TrainingPipeline:
    def __init__(self, config: TrainingPipelineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> list[dict[str, float | int | None]]:
        ingestion_config = DataIngestionConfig(
            data_dir=self.config.data_dir,
            artifacts_dir=str(Path(self.config.output_model_path).parent),
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )
        split_file_path = DataIngestion(ingestion_config).ingest()
        split_artifact = DataSplitArtifact.load(split_file_path)

        model = LeafDiseaseCNN(num_classes=len(split_artifact.class_names)).to(self.device)
        trainer = ModelTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            device=self.device,
            learning_rate=self.config.learning_rate,
        )

        history = trainer.train(split_file_path, self.config.epochs)
        self._save_model(model)
        return history

    def _save_model(self, model: torch.nn.Module) -> None:
        output_path = Path(self.config.output_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the banana leaf disease classifier")
    parser.add_argument("--data-dir", required=True, help="Path to the ImageFolder dataset root")
    parser.add_argument("--output-model-path", default="artifacts/model.pth", help="Path to save the trained model")
    parser.add_argument("--image-size", type=int, default=150, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingPipelineConfig(
        data_dir=args.data_dir,
        output_model_path=args.output_model_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
