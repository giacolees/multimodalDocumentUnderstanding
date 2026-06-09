FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --extra services --no-dev --frozen

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py ./shared/observability.py
COPY services/evaluation-svc/main.py ./main.py
COPY services/evaluation-svc/judge.py ./judge.py
COPY services/evaluation-svc/rag_scorer.py ./rag_scorer.py

EXPOSE 8003
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
