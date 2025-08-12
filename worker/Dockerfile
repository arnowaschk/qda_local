FROM python:3.11-slim

# Fix /tmp directory permissions and create it properly
RUN mkdir -p /tmp && chmod 1777 /tmp

# Create global cache directories (these will be mounted from host)
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /root/.local/share/spacy
RUN mkdir -p /root/.cache/torch
RUN mkdir -p /root/.cache/torch/sentence_transformers

# Set environment variables to use global cache locations
# Remove deprecated TRANSFORMERS_CACHE, use HF_HOME instead
ENV HF_HOME=/root/.cache/huggingface
ENV SPACY_DATA=/root/.local/share/spacy
ENV TORCH_HOME=/root/.cache/torch
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git wget curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements_torch.txt /app/requirements_torch.txt
RUN pip install --no-cache-dir -r requirements_torch.txt

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model to global cache
RUN python -c "import sys,subprocess; subprocess.call([sys.executable,'-m','spacy','download','de_core_news_lg'])"

COPY *.py /app

EXPOSE 8001

CMD ["python", "main.py"]
