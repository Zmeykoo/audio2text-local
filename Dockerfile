# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and system dependencies
# ffmpeg is required for pydub audio conversion
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt .

# Install PyTorch with CUDA support and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/

RUN mkdir -p /app/data

# Set environment variables for CUDA and Python
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda

CMD ["python", "main.py"]
