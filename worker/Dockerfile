# Single stage build for simplicity
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-setuptools python3-wheel \
    build-essential cmake gcc g++ git curl \
    libopenblas-dev libblas-dev liblapack-dev \
    ninja-build libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Add symlink for libcuda.so.1
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/root/.local/bin:$PATH" \
    HF_HOME="/cache/huggingface" \
    SPACY_DATA="/cache/spacy" \
    TORCH_HOME="/cache/torch" \
    SENTENCE_TRANSFORMERS_HOME="/cache/sentence-transformers" \
    TOKENIZERS_PARALLELISM=false \
    LOG_LEVEL=INFO

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data \
    /app/out \
    /app/config \
    /cache/huggingface \
    /cache/spacy \
    /cache/torch \
    /cache/sentence-transformers \
    && chmod -R 777 /app /cache


# Install Python dependencies

COPY requirements_torch.txt ./
RUN pip install --user -r requirements_torch.txt

COPY requirements.txt ./
RUN pip install --user -r requirements.txt

# Download spaCy model
RUN python -c "import spacy; spacy.cli.download('de_core_news_lg')"

COPY requirements_new.txt ./
RUN pip install --user -r requirements_new.txt

# Copy application code
#COPY  *.py /app/ # this is done as mount in docker compose

# Expose the port the app runs on
EXPOSE 8001

# Set Python to run in unbuffered mode and ensure proper signal handling
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Use a shell form of CMD to ensure signals are properly handled
# Using python -u to ensure unbuffered output
CMD exec python -u -m uvicorn main:app --host 0.0.0.0 --port 8001 --log-level info --reload
