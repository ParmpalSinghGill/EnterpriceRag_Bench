# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies (build-essential, git, libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY llamaindex_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application and core retrieval modules
COPY retrieval/ /app/retrieval/
COPY generation/ /app/generation/
COPY llamaindex_app/ /app/llamaindex_app/

# Expose HTTP port
EXPOSE 8000

# Healthcheck for AWS Application Load Balancer
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn server
CMD ["uvicorn", "llamaindex_app.app:app", "--host", "0.0.0.0", "--port", "8000"]

