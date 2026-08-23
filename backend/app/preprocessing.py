import numpy as np
from PIL import Image

from .config import Settings


def preprocess_image(image: Image.Image, settings: Settings) -> np.ndarray:
    converted = image.convert(settings.color_mode)
    resized = converted.resize((settings.image_width, settings.image_height), Image.Resampling.LANCZOS)
    batch = np.asarray(resized, dtype=np.float32)
    if settings.color_mode == "L":
        batch = np.expand_dims(batch, axis=-1)
    if settings.scale_mode == "zero_one":
        batch /= 255.0
    elif settings.scale_mode == "minus_one_one":
        batch = batch / 127.5 - 1.0
    mean = settings.channel_values(settings.normalization_mean)
    std = settings.channel_values(settings.normalization_std)
    if (mean is None) != (std is None):
        raise ValueError("NORMALIZATION_MEAN and NORMALIZATION_STD must be provided together.")
    if mean and std:
        channels = batch.shape[-1]
        if len(mean) != channels or len(std) != channels or any(value == 0 for value in std):
            raise ValueError("Normalization values must match image channels and standard deviations cannot be zero.")
        batch = (batch - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.expand_dims(batch, axis=0)
