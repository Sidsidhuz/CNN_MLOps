from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.components.artifacts import DataSplitArtifact
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.model import LeafDiseaseCNN
from src.components.trainer import ModelTrainer


@dataclass(frozen=True)
class TrainingPipelineConfig:
    data_dir: str
    output_model_path: str = "artifacts/model.pth"
    image_size: int = 150
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 42
    epochs: int = 25
    learning_rate: float = 1e-3


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