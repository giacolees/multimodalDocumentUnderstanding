FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27" \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/api-gateway/main.py /app/main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
