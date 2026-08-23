from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


async def read_validated_image(upload: UploadFile, max_upload_bytes: int) -> Image.Image:
    filename = upload.filename or ""
    suffix = filename.lower().rsplit(".", maxsplit=1)
    extension = f".{suffix[-1]}" if len(suffix) == 2 else ""
    if upload.content_type not in ALLOWED_CONTENT_TYPES or extension not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload a JPEG, PNG, or WebP image.")
    content = await upload.read(max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image exceeds the {max_upload_bytes // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(BytesIO(content)) as source:
            source.verify()
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from exc
    if image.width < 8 or image.height < 8:
        raise HTTPException(status_code=400, detail="Image dimensions must be at least 8 by 8 pixels.")
    return image
