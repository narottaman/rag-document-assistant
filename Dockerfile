# Dockerfile
# Runs on your laptop — CPU only, no GPU needed
# Uses Python 3.12 slim for small image size

FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy project
COPY . .

EXPOSE 7860 8000