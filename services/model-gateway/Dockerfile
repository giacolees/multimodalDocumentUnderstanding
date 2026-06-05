FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27"

COPY services/model-gateway/main.py /app/main.py
COPY services/model-gateway/pool.py /app/pool.py
COPY services/model-gateway/clients.py /app/clients.py

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
