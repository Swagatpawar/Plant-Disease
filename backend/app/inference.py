import json
from pathlib import Path
from threading import Lock

import numpy as np

from .config import Settings
from .schemas import PredictionItem, PredictionResponse


class ModelUnavailableError(RuntimeError):
    """Model artifacts cannot serve inference."""


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._labels: dict[int, str] = {}
        self._error: str | None = None
        self._lock = Lock()

    @property
    def ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None and bool(self._labels)

    @property
    def error(self) -> str | None:
        self._ensure_loaded()
        return self._error

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._error is not None:
            return
        with self._lock:
            if self._model is not None or self._error is not None:
                return
            try:
                self._load_artifacts()
            except Exception as exc:
                self._error = str(exc)

    def _load_artifacts(self) -> None:
        model_path = self.settings.model_file
        mapping_path = self.settings.class_mapping_file
        if not model_path.is_file():
            raise ModelUnavailableError(f"Model file not found: {model_path}")
        if not mapping_path.is_file():
            raise ModelUnavailableError(f"Class mapping file not found: {mapping_path}")
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        labels = _read_labels(mapping_path)
        output_classes = int(model.output_shape[-1])
        if output_classes != len(labels):
            raise ModelUnavailableError(f"Model outputs {output_classes} classes but mapping has {len(labels)} entries.")
        self._model, self._labels = model, labels

    def predict(self, batch: np.ndarray) -> PredictionResponse:
        if not self.ready:
            raise ModelUnavailableError(self.error or "Model is unavailable.")
        probabilities = np.asarray(self._model.predict(batch, verbose=0))[0]
        if probabilities.ndim != 1 or len(probabilities) != len(self._labels) or not np.isfinite(probabilities).all():
            raise ModelUnavailableError("Model returned invalid prediction values.")
        indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
        predictions = [PredictionItem(label=self._labels[int(index)], confidence=float(probabilities[index])) for index in indices]
        return PredictionResponse(prediction=predictions[0], top_predictions=predictions, low_confidence=predictions[0].confidence < self.settings.low_confidence_threshold, threshold=self.settings.low_confidence_threshold)


def _read_labels(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ModelUnavailableError("Class mapping must be a JSON object of index to label.")
    try:
        labels = {int(index): str(label) for index, label in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ModelUnavailableError("Class mapping keys must be integer indices.") from exc
    if not labels or sorted(labels) != list(range(len(labels))) or any(not value.strip() for value in labels.values()):
        raise ModelUnavailableError("Class mapping needs non-empty sequential indices starting at 0.")
    return labels
