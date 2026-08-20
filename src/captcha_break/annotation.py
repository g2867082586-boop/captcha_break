"""Resumable, local-only annotation workspace for authorized CAPTCHA images."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .data import IMAGE_SUFFIXES

MANIFEST_SCHEMA = "captcha-annotation/v1"


@dataclass(frozen=True, slots=True)
class Prelabel:
    beta_text: str
    default_text: str


@dataclass(slots=True)
class AnnotationRecord:
    filename: str
    sha256: str
    beta_prediction: str
    default_prediction: str
    suggested_label: str
    label: str | None = None
    status: str = "pending"
    updated_at_utc: str | None = None


class AnnotationWorkspace:
    def __init__(self, path: Path, *, alphabet: str, label_length: int = 4) -> None:
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must contain unique characters")
        if label_length != 4:
            raise ValueError("only four-character labels are supported")
        self.path = path.expanduser().resolve()
        self.images_dir = self.path / "images"
        self.manifest_path = self.path / "manifest.json"
        self.alphabet = alphabet
        self.label_length = label_length
        self.records: list[AnnotationRecord] = []
        self.source_dir: Path | None = None
        if self.manifest_path.exists():
            self._load()

    def _valid_label(self, value: str) -> bool:
        return len(value) == self.label_length and set(value) <= set(self.alphabet)

    def validate_label(self, value: str) -> str:
        normalized = value.strip().upper()
        if not self._valid_label(normalized):
            raise ValueError(
                f"label must contain exactly {self.label_length} characters from {self.alphabet}"
            )
        return normalized

    def _load(self) -> None:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("annotation manifest schema is unsupported")
        if payload.get("alphabet") != self.alphabet:
            raise ValueError("annotation manifest alphabet does not match")
        if payload.get("label_length") != self.label_length:
            raise ValueError("annotation manifest label length does not match")
        self.source_dir = Path(payload["source_dir"])
        self.records = [AnnotationRecord(**item) for item in payload["records"]]

    def _save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": MANIFEST_SCHEMA,
            "source_dir": str(self.source_dir),
            "alphabet": self.alphabet,
            "label_length": self.label_length,
            "records": [asdict(record) for record in self.records],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def prepare(self, source_dir: Path, predictor: Callable[[bytes], Prelabel]) -> int:
        source = source_dir.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"source image directory does not exist: {source}")
        if self.source_dir is not None and self.source_dir.resolve() != source:
            raise ValueError("existing workspace belongs to another source directory")
        self.source_dir = source
        self.path.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        existing = {record.filename: record for record in self.records}
        added = 0
        paths = sorted(
            item
            for item in source.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no supported images found in {source}")
        for image_path in paths:
            data = image_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if image_path.name in existing:
                if existing[image_path.name].sha256 != digest:
                    raise ValueError(
                        f"source image changed after annotation started: {image_path.name}"
                    )
                continue
            working_image = self.images_dir / image_path.name
            shutil.copy2(image_path, working_image)
            source_metadata = image_path.with_suffix(".json")
            if source_metadata.is_file():
                shutil.copy2(source_metadata, self.images_dir / source_metadata.name)
            prediction = predictor(data)
            beta = prediction.beta_text.strip().upper()
            default = prediction.default_text.strip().upper()
            suggestion = beta if self._valid_label(beta) else default
            if not self._valid_label(suggestion):
                suggestion = ""
            self.records.append(
                AnnotationRecord(
                    filename=image_path.name,
                    sha256=digest,
                    beta_prediction=beta,
                    default_prediction=default,
                    suggested_label=suggestion,
                )
            )
            added += 1
        self.records.sort(
            key=lambda item: (
                item.status != "pending",
                item.beta_prediction == item.default_prediction,
                item.filename,
            )
        )
        self._save()
        return added

    def progress(self) -> dict[str, int]:
        counts = {"pending": 0, "confirmed": 0, "skipped": 0}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return {"total": len(self.records), **counts}

    def state(self, index: int) -> dict[str, object]:
        if not self.records:
            raise ValueError("annotation workspace is empty")
        position = max(0, min(index, len(self.records) - 1))
        record = self.records[position]
        return {
            "index": position,
            "record": asdict(record),
            "progress": self.progress(),
        }

    def next_pending(self, current_index: int) -> int:
        if not self.records:
            return 0
        for offset in range(1, len(self.records) + 1):
            index = (current_index + offset) % len(self.records)
            if self.records[index].status == "pending":
                return index
        return max(0, min(current_index, len(self.records) - 1))

    def _update_metadata_label(self, record: AnnotationRecord) -> None:
        metadata_path = (self.images_dir / record.filename).with_suffix(".json")
        payload: dict[str, object] = {}
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["label"] = record.label
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)

    def confirm(self, index: int, label: str) -> int:
        record = self.records[index]
        record.label = self.validate_label(label)
        record.status = "confirmed"
        record.updated_at_utc = datetime.now(UTC).isoformat()
        self._update_metadata_label(record)
        self._save()
        return self.next_pending(index)

    def skip(self, index: int) -> int:
        record = self.records[index]
        record.label = None
        record.status = "skipped"
        record.updated_at_utc = datetime.now(UTC).isoformat()
        self._update_metadata_label(record)
        self._save()
        return self.next_pending(index)

    def export_confirmed(self) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = self.path / "exports" / timestamp
        output.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, str]] = []
        for index, record in enumerate(self.records, start=1):
            if record.status != "confirmed" or record.label is None:
                continue
            source = self.images_dir / record.filename
            destination = output / f"{record.label}_{index:04d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            rows.append(
                {
                    "filename": destination.name,
                    "label": record.label,
                    "source_filename": record.filename,
                    "sha256": record.sha256,
                }
            )
        with (output / "labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["filename", "label", "source_filename", "sha256"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return output
