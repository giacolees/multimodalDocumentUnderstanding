FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "redis[hiredis]>=5.0" \
    "redisvl>=0.2" \
    "sentence-transformers>=3.0" \
    "pillow>=10.0"

COPY services/document-svc/main.py /app/main.py
COPY services/document-svc/indexer.py /app/indexer.py
COPY services/document-svc/search.py /app/search.py

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
