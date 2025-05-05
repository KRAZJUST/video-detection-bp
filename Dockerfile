# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    x11-apps libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup virtualenv
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add non-root user to run GUI apps if needed
RUN useradd -ms /bin/bash devuser
USER devuser
WORKDIR /home/devuser/app
COPY . .

CMD ["python3", "src/project/main.py"]
