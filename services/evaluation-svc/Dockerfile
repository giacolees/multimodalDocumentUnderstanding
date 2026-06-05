FROM python:3.11-slim

WORKDIR /app

# Install deps directly (no uv needed in container)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    pydantic-ai>=1.0.0 \
    pydantic>=2.0 \
    pillow>=10.0 \
    requests>=2.31.0

# Copy src package (needed for LLMJudge and metrics)
COPY src/ /app/src/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e . --no-deps

# Copy service files
COPY services/evaluation-svc/main.py /app/main.py
COPY services/evaluation-svc/judge.py /app/judge.py
COPY services/evaluation-svc/rag_scorer.py /app/rag_scorer.py

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
