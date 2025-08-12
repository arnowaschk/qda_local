FROM python:3.11-slim

# Fix /tmp directory permissions and create it properly
RUN mkdir -p /tmp && chmod 1777 /tmp

# Now install packages including Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY app /app/app
EXPOSE 8000
