FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra services --no-dev --frozen --no-install-project

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py ./shared/observability.py
COPY services/api-gateway/main.py ./main.py

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
