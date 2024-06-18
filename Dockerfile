FROM python:3.9-slim

RUN useradd -m -u 1000 user

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model explicitly first
COPY models/rossmann_production_model.pkl /app/models/

# Copy rest of app
COPY --chown=user . .

RUN mkdir -p /app/logs && chown -R user:user /app/logs || true

ARG MODEL_VERSION=2.0.0
ENV MODEL_VERSION=${MODEL_VERSION}

USER user

EXPOSE 7860

ENV PYTHONPATH=/app
ENV PORT=7860

CMD exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}
