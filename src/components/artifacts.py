from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataSplitArtifact:
    data_dir: str
    class_names: list[str]
    train_indices: list[int]
    val_indices: list[int]
    image_size: int
    batch_size: int
    val_split: float
    seed: int

    def save(self, file_path: str | Path) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, file_path: str | Path) -> "DataSplitArtifact":
        path = Path(file_path)
        return cls(**json.loads(path.read_text(encoding="utf-8")))