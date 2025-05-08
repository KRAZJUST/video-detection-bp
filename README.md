# Video Processing & Search System
---
**Author:** David Skalka

**Date:** 2025-05-07

**Description:** This is the implementation of the system designed as a part of thesis `Recognizing People and Their Activities in Video from Security Cameras` at Faculty of Information Technology, Brno University of Technology.

---
This application is a high-performance, modular video processing pipeline designed for fast indexing, object detection, tracking, and intelligent querying of large-scale video datasets. It leverages state-of-the-art models like YOLO, ByteTrack, X-CLIP, and SigLIP to enable offline search by objects, actions, colors and directions in natural language queries. 

The system is divided into 3 logical processing levels - YOLO, YOLO+ByteTrack and X-CLIP/SigLIP

**Disclaimer:** GitHub Copilot autocomplete was used during the implementation of this system

---

## 📦 Features

- **⚡ Ultra-Fast Preprocessing**  
  up to ~30× real-time processing via optimized frame extraction with `ffmpeg` and usage of X-CLIP or YOLO models on NVIDIA GeForce RTX 2050 GPU. (slower when using other models)

- **🧠 Object Detection & Tracking**  
  - YOLOv11 for object detection (e.g., person, vehicle).  
  - ByteTrack for identity-preserving multi-object tracking.

- **🎨 Color & Directional Analysis**  
  - Extracts dominant colors per object (bounding box region with circular mask or segmentation mask).  
  - Estimates object motion direction (e.g., "north", "north-west").  

- **🧠 Vision-Language Embeddings**  
  - X-CLIP & SigLIP to embed video frames.  
  - Cosine similarity search over ChromaDB vector store.  
  - Supports complex natural language queries (e.g., `"a person waving near a red car"`).

- **🗃️ Embedded Storage & Query Engine**  
  - Uses SQLite3 and ChromaDB.  
  - Storage is reset on start of new video processing for that particular level.
  - Fast top-k retrieval via vector search.

- **🖼️ Flexible Query Modes**  
  - Post-indexing search or real-time filtering.  

- **🔍 Minimalistic UI**
  - UI for easy usage of the system without the need of CLI.
---

## 🛠️ Architecture Overview

TODO

- Documentation for each class and method can be found in [documentation](docs/build/html/index.html)

---

## 🚀 Setup
### System requirements
- Ubuntu (recommended 22.04 LTS or newer) or other Linux distro
- At least 4GB RAM (8GB+ recommended)
- 20GB free disk space (Maybe could be less)
- NVIDIA GPU (optional but recommended for better performance)

> TODO: add Docker setup

### 🧪 Virtual Environment Setup (for development or testing)
> This is example for Ubuntu and might be different on other distros
1. Install system dependencies
```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip \
    libxcb-cursor0 libxcb-xinerama0 libgl1-mesa-glx libglib2.0-0 \
    sqlite3 ffmpeg libsm6 libxrender1 libqt6core6 libqt6gui6 \
    libqt6widgets6 libxcb-xinerama0
```   

2. Create and activate virtual environment
```
cd ~/video-detection-bp
python3 -m venv venv
source venv/bin/activate
```
3. Install Python dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
4. Run the app
```
source venv/bin/activate (only if the virtual environment is not already activated)
python src/project/main.py
```

### ⚠️ Common Issues
**Missing Qt plugin 'xcb':** 
- Ensure libxcb-cursor0 and libxcb-xinerama0 are installed.

**Missing out correct nvidia drivers for GPU acceleration:** 
- Ensure correct CUDA and cuDNN versions installed
  - `nvidia-smi`
  - `nvcc --version`
- This project uses GPU acceleration by default. Ensure you have a compatible CUDA environment. If not, install the CPU-only versions of PyTorch and TensorFlow.


---
## 🏗️ Project Structure
The project should have the followin structure:
```
video-detection-bp/
├── docker-compose.yaml
├── Dockerfile
├── docs/
├── entrypoint.sh
├── README.md
├── requirements.txt
├── runs.zip
└── src/
    ├── examples/
    ├── outputs/
    └── project/
        ├── app_profile.prof
        ├── constants/
        ├── database/
        ├── database_operations.log
        ├── detections.db
        ├── detectors/
        ├── env/
        ├── interface/
        ├── main.py
        ├── parsers/
        ├── processors/
        ├── profile_graph.png
        ├── profiling_utils/
        ├── siglip/
        ├── vector_database/
        ├── xclip/
        ├── yolo11n.pt
        └── yolo11n-seg.pt
```
---

## 🔧 Configuration

- Frame skip interval: adjustable (e.g., every 10th frame for more stable tracking)  
- Detection categories: `['person', 'car', 'truck', ...]`  configurable only in code
- Processing model and depth of analysis (segmentation usage for color calculation)

---

## 🧪 Evaluation & Accuracy

- Each model accuraccy and speed evaluatioin
- Studying how can the SigLIP working with static frames handle queries designed for X-CLIP (with verbs)

### 📊 Benchmark Results

| **Method**       | **Avg. Accuracy [%]** | **Avg. RTF (× real-time)** |
|------------------|-----------------------|-----------------------------|
| **YOLO-seg**      | 80.94%                | 12.69×                      |
| **YOLO**          | 78.53%                | 32.29×                      |
| **ByteTrack**     | 69.00%                | 18.67×                      |
| **X-CLIP**        |                       | 32.60×                      |
| ├─ Hit@10         | 56.92%                |                             |
| ├─ Hit@5          | 49.80%                |                             |
| ├─ Hit@3          | 37.30%                |                             |
| └─ Hit@1          | 23.51%                |                             |
| **SigLIP***       | Hit@80: 45.45%        | 17.80×                      |
| **SigLIP**        |                       | 20.17×                      |
| ├─ Hit@80         | 43.87%                |                             |
| └─ Hit@40         | 35.18%                |                             |

*Note-1: RTF = Real-Time Factor. Results averaged across all resolutions.*
*Note-2: SigLIP\* stands for SigLIP model processing only frames with detections filtered by YOLO detections.*
*Note-3: YOLO-seg stands for classic YOLO model used for initial detections and it's segmentation model used for color calculation.*

---
