#!/bin/bash

# Ensure NVIDIA drivers are installed first
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not installed. Exiting."
    exit 1
fi

# Clean up old setups
sudo apt-get purge -y nvidia-docker nvidia-docker2 nvidia-container-toolkit

# Set up NVIDIA container toolkit
# Always fallback to supported distro if detection fails
distribution=$(. /etc/os-release; echo "${ID}22.04")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
