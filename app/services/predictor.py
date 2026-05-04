from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import UploadFile
from PIL import Image
from torchvision import transforms

from src.components.model import LeafDiseaseCNN


@dataclass(frozen=True)
class PredictionResult:
    crop: str
    disease: str
    confidence: float


class CropPredictor:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def available_crops(self) -> list[str]:
        crops = set()
        for model_path in self.base_dir.glob("*_model.pth"):
            crops.add(model_path.stem.replace("_model", ""))
        return sorted(crops)

    def _metadata_path(self, crop: str) -> Path:
        return self.base_dir / f"{crop}_model_metadata.json"

    def _model_path(self, crop: str) -> Path:
        return self.base_dir / f"{crop}_model.pth"

    def _load_metadata(self, crop: str) -> dict[str, Any]:
        metadata_path = self._metadata_path(crop)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for crop '{crop}' at {metadata_path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _load_model(self, crop: str, num_classes: int) -> LeafDiseaseCNN:
        model_path = self._model_path(crop)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for crop '{crop}' at {model_path}")

        model = LeafDiseaseCNN(num_classes=num_classes, pretrained=False).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _transform(self, image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    async def predict(self, crop: str, uploaded_file: UploadFile) -> PredictionResult:
        metadata = self._load_metadata(crop)
        model = self._load_model(crop, num_classes=int(metadata["num_classes"]))

        image_bytes = await uploaded_file.read()
        if not image_bytes:
            raise ValueError("Uploaded image is empty")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        transform = self._transform(int(metadata["image_size"]))
        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        class_names = metadata["class_names"]
        predicted_class = class_names[int(predicted_index.item())]
        return PredictionResult(
            crop=crop,
            disease=predicted_class,
            confidence=float(confidence.item() * 100.0),
        )
