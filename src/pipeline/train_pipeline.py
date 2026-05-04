from __future__ import annotations

from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline, DataIngestionPipelineConfig
from src.pipeline.model_training_pipeline import ModelTrainingPipeline, ModelTrainingPipelineConfig


class TrainingPipeline:
    def __init__(self, data_dir: str, output_model_path: str, image_size: int, batch_size: int, val_split: float, seed: int, epochs: int, learning_rate: float) -> None:
        self.data_dir = data_dir
        self.output_model_path = output_model_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate

    def run(self) -> list[dict[str, float | int | None]]:
        split_path = DataIngestionPipeline(
            DataIngestionPipelineConfig(
                data_dir=self.data_dir,
                artifacts_dir="artifacts",
                image_size=self.image_size,
                batch_size=self.batch_size,
                val_split=self.val_split,
                seed=self.seed,
            )
        ).run()

        return ModelTrainingPipeline(
            ModelTrainingPipelineConfig(
                split_file_path=split_path,
                output_model_path=self.output_model_path,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
            )
        ).run()