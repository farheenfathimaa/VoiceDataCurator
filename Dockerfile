# ──────────────────────────────────────────────
#  VoiceDataCurator — Dockerfile
#  Builds a container that runs the pipeline CLI
# ──────────────────────────────────────────────
FROM python:3.11-slim

# Install system deps: ffmpeg (required by Whisper + librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create runtime directories
RUN mkdir -p data/raw output rejected logs

# Expose Streamlit port
EXPOSE 8501

# Default: run the CLI pipeline
# Override with: docker run ... streamlit run dashboard.py
CMD ["python", "main.py", "--input", "./data/raw", "--output", "./output"]
