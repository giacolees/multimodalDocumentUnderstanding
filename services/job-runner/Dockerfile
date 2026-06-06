FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27" \
    "redis[hiredis]>=5.0" \
    "pyyaml>=6.0" \
    "pillow>=10.0" \
    "requests>=2.31.0" \
    "pydantic-ai>=1.0.0" \
    "langchain-core>=0.3.0" \
    "datasets>=2.18.0" \
    "pandas>=2.0" \
    "pdf2image>=1.17" \
    "pypdf>=4.0" \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

RUN pip install --no-cache-dir spacy>=3.7 && \
    python -m spacy download en_core_web_sm

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e . --no-deps

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/job-runner/main.py /app/main.py
COPY services/job-runner/state.py /app/state.py
COPY services/job-runner/jobs/ /app/jobs/

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
