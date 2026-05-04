from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.components.artifacts import DataSplitArtifact


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        device: torch.device,
        learning_rate: float = 1e-3,
        criterion: nn.Module | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=learning_rate)
        print(f"[ModelTrainer.__init__] device={self.device}", flush=True)

    def _build_transforms(self, image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        return train_transform, val_transform

    def _create_dataloaders(self, split_file_path: str) -> tuple[DataLoader, DataLoader]:
        print(f"[ModelTrainer._create_dataloaders] loading split file {split_file_path}", flush=True)
        artifact = DataSplitArtifact.load(split_file_path)
        train_transform, val_transform = self._build_transforms(artifact.image_size)

        train_dataset = datasets.ImageFolder(root=artifact.data_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=artifact.data_dir, transform=val_transform)

        print(
            f"[ModelTrainer._create_dataloaders] train_indices={len(artifact.train_indices)} val_indices={len(artifact.val_indices)}",
            flush=True,
        )

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            Subset(train_dataset, artifact.train_indices),
            batch_size=artifact.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            Subset(val_dataset, artifact.val_indices),
            batch_size=artifact.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def _run_epoch(self, dataloader: DataLoader, training: bool, epoch_index: int, num_epochs: int) -> EpochMetrics:
        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if training else torch.no_grad()
        phase_name = "train" if training else "val"
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch_index}/{num_epochs}] {phase_name}",
            leave=False,
            ascii=True,
            dynamic_ncols=True,
        )

        with context:
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += batch_size

                progress_bar.set_postfix(
                    loss=f"{(running_loss / total):.4f}" if total else "0.0000",
                    acc=f"{((correct / total) * 100):.2f}%" if total else "0.00%",
                )

        average_loss = running_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        return EpochMetrics(loss=average_loss, accuracy=accuracy)

    def train(self, split_file_path: str, num_epochs: int) -> list[dict[str, Any]]:
        print("[ModelTrainer.train] starting training loop", flush=True)
        train_loader, val_loader = self._create_dataloaders(split_file_path)
        history: list[dict[str, Any]] = []

        for epoch in range(num_epochs):
            print(f"[ModelTrainer.train] epoch {epoch + 1}/{num_epochs} started", flush=True)
            train_metrics = self._run_epoch(train_loader, training=True, epoch_index=epoch + 1, num_epochs=num_epochs)
            val_metrics = (
                self._run_epoch(val_loader, training=False, epoch_index=epoch + 1, num_epochs=num_epochs)
                if val_loader is not None
                else None
            )

            if val_metrics is not None:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {train_metrics.loss:.4f} "
                    f"Train Acc: {train_metrics.accuracy * 100:.2f}% "
                    f"Val Loss: {val_metrics.loss:.4f} "
                    f"Val Acc: {val_metrics.accuracy * 100:.2f}%"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Loss: {train_metrics.loss:.4f} "
                    f"Accuracy: {train_metrics.accuracy * 100:.2f}%"
                )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "val_loss": val_metrics.loss if val_metrics is not None else None,
                    "val_accuracy": val_metrics.accuracy if val_metrics is not None else None,
                }
            )

        print("[ModelTrainer.train] training loop complete", flush=True)

        return history
