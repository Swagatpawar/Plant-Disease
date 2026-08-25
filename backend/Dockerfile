FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY backend ./backend
COPY model ./model
# Download the trained model from Hugging Face.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/Swagatpawar/plantguard-ai-model/resolve/main/plant_disease_model_final.keras?download=true', '/app/model/plant_disease_model.keras')"

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
