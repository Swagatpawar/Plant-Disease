import numpy as np
from PIL import Image

from backend.app.config import Settings
from backend.app.preprocessing import preprocess_image


def test_zero_one_preprocessing_has_expected_shape_and_range():
    settings = Settings(image_width=4, image_height=3, scale_mode="zero_one")
    image = Image.new("RGB", (8, 8), color=(255, 127, 0))

    batch = preprocess_image(image, settings)

    assert batch.shape == (1, 3, 4, 3)
    assert np.isclose(batch[0, 0, 0, 0], 1.0)
    assert np.isclose(batch[0, 0, 0, 1], 127 / 255)


def test_minus_one_one_preprocessing():
    settings = Settings(image_width=2, image_height=2, scale_mode="minus_one_one")
    batch = preprocess_image(Image.new("RGB", (2, 2), color="black"), settings)

    assert np.allclose(batch, -1.0)
