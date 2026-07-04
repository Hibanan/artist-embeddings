# RunPod Docker image for artist embeddings training
# Build: docker build -t artist-embeddings .
# Run:   docker run --gpus all artist-embeddings

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer cache: seldom changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (layer cache: changes every build)
COPY . .

# RunPod serves on 8888 by default; we don't need a server,
# but the port must be open for the pod to stay alive.
EXPOSE 8888

# HuggingFace cache goes to a persistent volume to survive pod restarts
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache

CMD ["bash", "runpod_start.sh"]
