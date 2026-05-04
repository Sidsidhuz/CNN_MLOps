from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from src.components.artifacts import DataSplitArtifact
from src.components.model import LeafDiseaseCNN
from src.components.trainer import ModelTrainer


@dataclass(frozen=True)
class ModelTrainingPipelineConfig:
    split_file_path: str
    output_model_path: str = "artifacts/model.pth"
    epochs: int = 25
    learning_rate: float = 1e-3


class ModelTrainingPipeline:
    def __init__(self, config: ModelTrainingPipelineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelTrainingPipeline.__init__] device={self.device}", flush=True)

    def run(self) -> list[dict[str, float | int | None]]:
        print("[ModelTrainingPipeline.run] loading split artifact", flush=True)
        split_artifact = DataSplitArtifact.load(self.config.split_file_path)

        print(
            f"[ModelTrainingPipeline.run] creating LeafDiseaseCNN with num_classes={len(split_artifact.class_names)}",
            flush=True,
        )
        model = LeafDiseaseCNN(num_classes=len(split_artifact.class_names)).to(self.device)
        trainer = ModelTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            device=self.device,
            learning_rate=self.config.learning_rate,
        )

        print("[ModelTrainingPipeline.run] invoking ModelTrainer.train()", flush=True)
        history = trainer.train(self.config.split_file_path, self.config.epochs)
        self._save_model(model)
        print("[ModelTrainingPipeline.run] complete", flush=True)
        return history

    def _save_model(self, model: torch.nn.Module) -> None:
        output_path = Path(self.config.output_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model training stage")
    parser.add_argument("--split-file-path", required=True, help="Path to the split metadata file")
    parser.add_argument("--output-model-path", default="artifacts/model.pth", help="Path to save the model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ModelTrainingPipeline(
        ModelTrainingPipelineConfig(
            split_file_path=args.split_file_path,
            output_model_path=args.output_model_path,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    )
    pipeline.run()


if __name__ == "__main__":
    main()
