# PlantGuard AI

PlantGuard AI is a plant disease classification system: a FastAPI inference API backed by a TensorFlow/Keras model and a polished Streamlit interface. Drop in your trained model and class mapping — the system validates, preprocesses, runs inference, and returns ranked predictions.

## Features

- FastAPI `/health` and `/predict` endpoints
- **Eager model loading at startup** — `/health` is immediately accurate
- Strict upload validation (content type, extension, image decode, dimensions, 10 MB limit)
- Configurable inference preprocessing to preserve training compatibility
- TensorFlow/Keras `.keras` model loading with clear startup logging
- Class mapping validation and top-three confidence-ranked predictions
- Streamlit upload, preview, diagnosis, confidence badge (high/moderate/low), backend status, session history
- Unit tests for validation, preprocessing, class mapping, model loading, and response shaping

## Project layout

```text
backend/       FastAPI application and inference services
frontend/      Streamlit dashboard
model/         Trained model and class mapping (not committed to source control)
tests/         Backend and integration tests
```

## Quick start (Windows — no venv activation required)

> **Note:** PowerShell execution policy may block `Activate.ps1`. Use the direct Python path below instead.

### 1. Create the virtual environment and install dependencies

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### 2. Copy the environment template

```powershell
copy .env.example .env
```

Edit `.env` if you need to change ports, CORS origins, or preprocessing defaults.

### 3. Place model artifacts

Copy your trained model and class mapping into the `model/` folder:

```
model/plant_disease_model.keras
model/class_mapping.json
```

### 4. Start the backend API

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

The model loads at startup. Watch the console for:

```
PlantGuard AI — loading model artifacts…
Model loaded and ready.
```

Then verify: http://127.0.0.1:8000/health  
Interactive docs: http://127.0.0.1:8000/docs

### 5. Start the frontend (separate terminal)

```powershell
.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

Open http://localhost:8501

## Model integration

PlantGuard accepts Keras `.keras` models. The class mapping must be a JSON object from string integer index to label:

```json
{
  "0": "Apple Scab Leaf",
  "1": "Apple leaf",
  "2": "Apple rust leaf"
}
```

The number of mapping entries must equal the model's output classes. If they differ, the API refuses to start inference and reports the mismatch clearly.

Configure `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `SCALE_MODE`, `COLOR_MODE`, and optionally `NORMALIZATION_MEAN`/`NORMALIZATION_STD` in `.env` to match your training pipeline exactly.

| `SCALE_MODE` | Effect |
|---|---|
| `zero_one` | divide by 255 (default) |
| `minus_one_one` | `pixel / 127.5 - 1.0` |
| `none` | no scaling |

Until both artifacts exist, `/health` reports `model_ready: false`, `/predict` returns a clear 503, and the UI explains what to add.

## Confidence thresholds

| Confidence | UI indicator |
|---|---|
| ≥ 80 % | ✓ High confidence (green) |
| 50 – 79 % | ⚠ Moderate confidence (amber) |
| < 50 % | ⚠ Low confidence (red) |

Set `LOW_CONFIDENCE_THRESHOLD` in `.env` to control the model's own low-confidence flag (used separately from the UI tiers).

## Tests and checks

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
.venv\Scripts\python.exe -m ruff check backend frontend tests
```

Model-loading tests are skipped automatically if `plant_disease_model.keras` is not present.

## Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `PlantGuard AI API` | FastAPI title |
| `CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated allowed origins |
| `MODEL_PATH` | `model/plant_disease_model.keras` | Relative or absolute path to model |
| `CLASS_MAPPING_PATH` | `model/class_mapping.json` | Relative or absolute path to mapping |
| `IMAGE_WIDTH` | `224` | Resize target width |
| `IMAGE_HEIGHT` | `224` | Resize target height |
| `COLOR_MODE` | `RGB` | `RGB` or `L` (grayscale) |
| `SCALE_MODE` | `zero_one` | `zero_one`, `minus_one_one`, or `none` |
| `NORMALIZATION_MEAN` | *(blank)* | Comma-separated per-channel mean |
| `NORMALIZATION_STD` | *(blank)* | Comma-separated per-channel std |
| `LOW_CONFIDENCE_THRESHOLD` | `0.60` | Flag low-confidence predictions |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Frontend → backend URL |

## Deployment

Set `CORS_ORIGINS` to the deployed frontend origin(s), set `API_BASE_URL` for the UI, and deliver the trained model through your secure artifact workflow — do not commit model weights to source control.

For containers:

```powershell
docker compose up --build
```
