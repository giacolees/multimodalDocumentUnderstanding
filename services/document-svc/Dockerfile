FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra services --no-dev --frozen --no-install-project

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py ./shared/observability.py
COPY services/document-svc/main.py ./main.py
COPY services/document-svc/indexer.py ./indexer.py
COPY services/document-svc/search.py ./search.py

EXPOSE 8002
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
