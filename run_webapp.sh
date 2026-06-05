#!/bin/bash

# Define paths
ROOT_DIR=$(pwd)
FRONTEND_DIR="$ROOT_DIR/frontend"

# Check if environment is activated or python is available
if [ -d "$ROOT_DIR/.env" ]; then
    PYTHON_EXEC="$ROOT_DIR/.env/bin/python"
    UVICORN_EXEC="$ROOT_DIR/.env/bin/uvicorn"
else
    PYTHON_EXEC="python3"
    UVICORN_EXEC="uvicorn"
fi

echo "Starting NeuroVision Web App..."

# Kill existing processes on exit
trap 'kill %1; kill %2' EXIT

# Start FastAPI Backend
echo "Starting FastAPI backend on port 8000..."
cd $ROOT_DIR
$UVICORN_EXEC src.project.api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start Vite Frontend
echo "Starting Vite frontend on port 5173..."
cd $FRONTEND_DIR
npm run dev &
FRONTEND_PID=$!

echo "========================================================"
echo "NeuroVision Web Application is running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "========================================================"
echo "Press Ctrl+C to stop both servers."

wait
