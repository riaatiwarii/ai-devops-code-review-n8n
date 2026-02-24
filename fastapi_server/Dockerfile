FROM python:3.11-slim

WORKDIR /app

# System deps
# NEW
RUN apt-get update && apt-get install -y \
    curl git \
    && rm -rf /var/lib/apt/lists/*
# ADD THIS LINE before COPY requirements.txt
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]