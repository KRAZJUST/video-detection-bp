#!/bin/bash

# Verify display is accessible
echo "Display set to: $DISPLAY"
xhost +local:root || echo "Warning: xhost command failed, X11 forwarding might not work"

# Check NVIDIA GPU availability
nvidia-smi || echo "Warning: NVIDIA GPU not detected or drivers not properly configured"

# Check CUDA availability with a simple Python script
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count())" || echo "Warning: CUDA check failed"

# Run the application
cd /app
uvicorn src.project.api.main:app --host [IP_ADDRESS] --port 8000 --reload --log-level debug

# Keep container running if the app crashes (for debugging)
# exec "$@"