from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "PlantGuard AI API"
    environment: str = "development"
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    max_upload_bytes: int = 10 * 1024 * 1024
    model_path: str = "model/plant_disease_model.keras"
    class_mapping_path: str = "model/class_mapping.json"
    image_width: int = 240
    image_height: int = 240
    color_mode: str = "RGB"
    # The bundled EfficientNetB1 model contains its own 1/255 rescaling layer.
    scale_mode: str = "none"
    normalization_mean: str = ""
    normalization_std: str = ""
    low_confidence_threshold: float = 0.60

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    @field_validator("color_mode")
    @classmethod
    def validate_color_mode(cls, value: str) -> str:
        value = value.upper()
        if value not in {"RGB", "L"}:
            raise ValueError("COLOR_MODE must be RGB or L")
        return value

    @field_validator("scale_mode")
    @classmethod
    def validate_scale_mode(cls, value: str) -> str:
        value = value.lower()
        if value not in {"zero_one", "minus_one_one", "none"}:
            raise ValueError("SCALE_MODE must be zero_one, minus_one_one, or none")
        return value

    @property
    def model_file(self) -> Path:
        return _resolve_path(self.model_path)

    @property
    def class_mapping_file(self) -> Path:
        return _resolve_path(self.class_mapping_path)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @staticmethod
    def channel_values(raw: str) -> list[float] | None:
        if not raw.strip():
            return None
        return [float(value.strip()) for value in raw.split(",")]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
