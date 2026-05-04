from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion, DataIngestionConfig


@dataclass(frozen=True)
class DataIngestionPipelineConfig:
    data_dir: str
    artifacts_dir: str = "artifacts"
    split_file_name: str = "data_split.json"
    image_size: int = 150
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 42


class DataIngestionPipeline:
    def __init__(self, config: DataIngestionPipelineConfig) -> None:
        self.config = config

    def run(self) -> str:
        print("[DataIngestionPipeline.run] starting", flush=True)
        ingestion_config = DataIngestionConfig(
            data_dir=self.config.data_dir,
            artifacts_dir=self.config.artifacts_dir,
            split_file_name=self.config.split_file_name,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )
        split_file_path = DataIngestion(ingestion_config).ingest()
        print(f"[DataIngestionPipeline.run] created split file at {split_file_path}", flush=True)
        return split_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data ingestion stage")
    parser.add_argument("--data-dir", required=True, help="Path to the ImageFolder dataset root")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to write split metadata")
    parser.add_argument("--split-file-name", default="data_split.json", help="Name of the split metadata file")
    parser.add_argument("--image-size", type=int, default=150, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DataIngestionPipeline(
        DataIngestionPipelineConfig(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            split_file_name=args.split_file_name,
            image_size=args.image_size,
            batch_size=args.batch_size,
            val_split=args.val_split,
            seed=args.seed,
        )
    )
    pipeline.run()


if __name__ == "__main__":
    main()
