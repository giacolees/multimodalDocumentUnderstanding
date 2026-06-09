FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra services --no-dev --frozen --no-install-project

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py ./shared/observability.py
COPY services/model-gateway/main.py ./main.py
COPY services/model-gateway/pool.py ./pool.py
COPY services/model-gateway/clients.py ./clients.py

EXPOSE 8001
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
