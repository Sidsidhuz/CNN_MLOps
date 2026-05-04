from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.components.artifacts import DataSplitArtifact
from src.components.model import LeafDiseaseCNN
from src.components.trainer import ModelTrainer


@dataclass(frozen=True)
class ModelTrainingPipelineConfig:
    split_file_path: str
    output_model_path: str | None = None
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
        self._save_model(model, split_artifact)
        print("[ModelTrainingPipeline.run] complete", flush=True)
        return history

    def _resolve_output_model_path(self, split_artifact: DataSplitArtifact) -> Path:
        if self.config.output_model_path:
            return Path(self.config.output_model_path)

        crop_name = Path(split_artifact.data_dir).name.strip().lower().replace(" ", "_")
        crop_name = "".join(character if character.isalnum() or character == "_" else "_" for character in crop_name)
        return Path("artifacts") / f"{crop_name}_model.pth"

    def _resolve_metadata_path(self, output_model_path: Path) -> Path:
        return output_model_path.with_name(f"{output_model_path.stem}_metadata.json")

    def _save_model(self, model: torch.nn.Module, split_artifact: DataSplitArtifact) -> None:
        output_path = self._resolve_output_model_path(split_artifact)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)

        metadata_path = self._resolve_metadata_path(output_path)
        metadata = {
            "crop_name": Path(split_artifact.data_dir).name,
            "class_names": split_artifact.class_names,
            "image_size": split_artifact.image_size,
            "batch_size": split_artifact.batch_size,
            "num_classes": len(split_artifact.class_names),
            "model_name": output_path.stem,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model training stage")
    parser.add_argument("--split-file-path", required=True, help="Path to the split metadata file")
    parser.add_argument(
        "--output-model-path",
        default=None,
        help="Optional explicit model path. If omitted, saves as artifacts/<crop>_model.pth",
    )
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
