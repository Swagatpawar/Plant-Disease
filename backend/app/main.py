import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .image_utils import read_validated_image
from .inference import ModelService, ModelUnavailableError
from .preprocessing import preprocess_image
from .schemas import HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    settings = settings or get_settings()
    service = service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Eagerly load the model at startup so the first /health request
        # does not have to wait for TensorFlow initialisation (~20 s on CPU).
        logger.info("PlantGuard AI — loading model artifacts…")
        service._ensure_loaded()  # noqa: SLF001
        if service.ready:
            logger.info("Model loaded and ready.")
        else:
            logger.warning("Model NOT ready: %s", service.error)
        yield

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        ready = service.ready
        return HealthResponse(
            status="ok" if ready else "degraded",
            model_ready=ready,
            model_path=str(settings.model_file),
            detail=service.error,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(image: UploadFile = File(...)) -> PredictionResponse:
        if not service.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=service.error or "Model is unavailable.",
            )
        parsed_image = await read_validated_image(image, settings.max_upload_bytes)
        try:
            return service.predict(preprocess_image(parsed_image, settings))
        except ModelUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

    return app


app = create_app()
