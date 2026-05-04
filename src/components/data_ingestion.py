from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets

from src.components.artifacts import DataSplitArtifact


@dataclass(frozen=True)
class DataIngestionConfig:
    data_dir: str
    artifacts_dir: str = "artifacts"
    split_file_name: str = "data_split.json"
    image_size: int = 150
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        print(f"[DataIngestion.__init__] data_dir={self.config.data_dir}", flush=True)

    def ingest(self) -> str:
        print("[DataIngestion.ingest] starting ingestion", flush=True)
        data_path = Path(self.config.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

        print(f"[DataIngestion.ingest] loading ImageFolder from {data_path}", flush=True)
        base_dataset = datasets.ImageFolder(root=str(data_path))
        if not base_dataset.classes:
            raise ValueError(f"No classes found in dataset directory: {data_path}")

        dataset_size = len(base_dataset)
        val_size = int(dataset_size * self.config.val_split)
        train_size = dataset_size - val_size

        print(
            f"[DataIngestion.ingest] dataset_size={dataset_size} train_size={train_size} val_size={val_size}",
            flush=True,
        )

        generator = torch.Generator().manual_seed(self.config.seed)
        train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

        artifact = DataSplitArtifact(
            data_dir=str(data_path),
            class_names=list(base_dataset.classes),
            train_indices=list(train_subset.indices),
            val_indices=list(val_subset.indices),
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )

        artifact_path = Path(self.config.artifacts_dir) / self.config.split_file_name
    print(f"[DataIngestion.ingest] saving split artifact to {artifact_path}", flush=True)
        return artifact.save(artifact_path)
