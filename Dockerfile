FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc git curl ca-certificates \
      libpq-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY ./src /app/src
COPY ./models /app/models
COPY ./config /app/config

EXPOSE 8000

ENV PORT=8000

CMD ["uvicorn", "src.RoadGuardianAI.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
