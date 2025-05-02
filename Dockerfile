# Use CUDA 11.8 as it's compatible with both PyTorch and TensorFlow
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install recent SQLite
WORKDIR /tmp
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450100.tar.gz \
    && tar xvfz sqlite-autoconf-3450100.tar.gz \
    && cd sqlite-autoconf-3450100 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3450100* \
    && ldconfig

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    # Clean pip cache
    rm -rf /root/.cache/pip/*

# Copy application code
COPY . .

# Default command
CMD ["python3", "main.py"]