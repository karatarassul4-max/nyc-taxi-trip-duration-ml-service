FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "taxi_duration.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
