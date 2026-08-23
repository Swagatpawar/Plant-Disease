"""Tests for class mapping loading and validation."""

import json
from pathlib import Path

import pytest

from backend.app.config import PROJECT_ROOT
from backend.app.inference import ModelUnavailableError, _read_labels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mapping(data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests against the actual class_mapping.json
# ---------------------------------------------------------------------------

MAPPING_PATH = PROJECT_ROOT / "model" / "class_mapping.json"


@pytest.mark.skipif(not MAPPING_PATH.is_file(), reason="class_mapping.json not present")
def test_actual_class_mapping_loads():
    labels = _read_labels(MAPPING_PATH)
    assert isinstance(labels, dict)


@pytest.mark.skipif(not MAPPING_PATH.is_file(), reason="class_mapping.json not present")
def test_actual_class_mapping_has_15_classes():
    labels = _read_labels(MAPPING_PATH)
    assert len(labels) == 15, f"Expected 15 classes, got {len(labels)}"


@pytest.mark.skipif(not MAPPING_PATH.is_file(), reason="class_mapping.json not present")
def test_actual_class_mapping_keys_are_sequential():
    labels = _read_labels(MAPPING_PATH)
    assert sorted(labels.keys()) == list(range(len(labels)))


@pytest.mark.skipif(not MAPPING_PATH.is_file(), reason="class_mapping.json not present")
def test_actual_class_mapping_values_are_non_empty_strings():
    labels = _read_labels(MAPPING_PATH)
    for idx, label in labels.items():
        assert isinstance(label, str) and label.strip(), (
            f"Label at index {idx} is empty or not a string"
        )


# ---------------------------------------------------------------------------
# Validation behaviour with synthetic mappings
# ---------------------------------------------------------------------------

def test_valid_mapping_parses_correctly(tmp_path):
    data = {"0": "Apple Scab", "1": "Healthy", "2": "Rust"}
    labels = _read_labels(_write_mapping(data, tmp_path))
    assert labels == {0: "Apple Scab", 1: "Healthy", 2: "Rust"}


def test_non_sequential_keys_raise(tmp_path):
    data = {"0": "Healthy", "2": "Rust"}  # missing index 1
    with pytest.raises(ModelUnavailableError, match="sequential"):
        _read_labels(_write_mapping(data, tmp_path))


def test_empty_label_raises(tmp_path):
    data = {"0": "Healthy", "1": "  "}
    with pytest.raises(ModelUnavailableError):
        _read_labels(_write_mapping(data, tmp_path))


def test_non_integer_key_raises(tmp_path):
    data = {"apple": "Apple Scab", "healthy": "Healthy"}
    with pytest.raises(ModelUnavailableError):
        _read_labels(_write_mapping(data, tmp_path))


def test_empty_mapping_raises(tmp_path):
    with pytest.raises(ModelUnavailableError):
        _read_labels(_write_mapping({}, tmp_path))


def test_non_dict_mapping_raises(tmp_path):
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(["Apple", "Healthy"]), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        _read_labels(p)
