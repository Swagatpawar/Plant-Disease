import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from backend.app.image_utils import read_validated_image


def jpeg_upload() -> UploadFile:
    payload = BytesIO()
    Image.new("RGB", (16, 16), color="green").save(payload, format="JPEG")
    return UploadFile(filename="leaf.jpg", file=BytesIO(payload.getvalue()), headers={"content-type": "image/jpeg"})


def test_valid_image_is_decoded():
    image = asyncio.run(read_validated_image(jpeg_upload(), 1024 * 1024))
    assert image.size == (16, 16)


def test_invalid_extension_is_rejected():
    upload = UploadFile(filename="leaf.txt", file=BytesIO(b"not an image"), headers={"content-type": "text/plain"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(read_validated_image(upload, 1024))
    assert exc.value.status_code == 415
