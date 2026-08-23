"""Tests for model loading and output/class-count consistency."""

import pytest

from backend.app.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "model" / "plant_disease_model.keras"
MAPPING_PATH = PROJECT_ROOT / "model" / "class_mapping.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason="plant_disease_model.keras not present — skipping model loading tests",
)


@pytest.fixture(scope="module")
def loaded_model():
    import tensorflow as tf  # imported lazily to avoid slow TF init for other test modules

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def test_model_loads_without_error(loaded_model):
    assert loaded_model is not None


def test_model_input_shape(loaded_model):
    # EfficientNetB1 model: RGB 240×240
    shape = loaded_model.input_shape
    assert shape == (None, 240, 240, 3), f"Unexpected input shape: {shape}"


def test_model_output_shape(loaded_model):
    shape = loaded_model.output_shape
    assert len(shape) == 2, "Output should be 2-D (batch, classes)"
    assert shape[-1] == 15, f"Expected 15 output classes, got {shape[-1]}"


@pytest.mark.skipif(not MAPPING_PATH.is_file(), reason="class_mapping.json not present")
def test_model_output_matches_class_mapping(loaded_model):
    import json

    labels = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    model_classes = loaded_model.output_shape[-1]
    mapping_classes = len(labels)
    assert model_classes == mapping_classes, (
        f"MISMATCH: model outputs {model_classes} classes "
        f"but class_mapping.json has {mapping_classes} entries. "
        "Predictions would be incorrect — fix the mapping before running inference."
    )
