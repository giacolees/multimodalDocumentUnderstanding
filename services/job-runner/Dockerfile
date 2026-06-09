FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --extra services --extra nlp --no-dev --frozen
RUN uv run python -m spacy download en_core_web_sm

COPY configs/ ./configs/
RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py ./shared/observability.py
COPY services/job-runner/main.py ./main.py
COPY services/job-runner/state.py ./state.py
COPY services/job-runner/jobs/ ./jobs/

EXPOSE 8004
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
