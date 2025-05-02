# Video Processing & Search System

This application is a high-performance, modular video processing pipeline designed for fast indexing, object detection, tracking, and intelligent querying of large-scale video datasets. It leverages state-of-the-art models like YOLO, ByteTrack, DeepSORT, X-CLIP, and SigLIP to enable offline search by objects, actions, colors and directions in natural language queries. The system is divided
into 3 logical processing levels - YOLO, YOLO+ByteTrack and X-CLIP/SigLIP

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

---

## 🚀 Setup

1. create new virtual python environment using `python3 -m venv venv` in project directory
2. Activate the environment using `source venv/bin/activate`
3. Install dependencies using `pip install -r ../requirements.txt`
4. Run program using `python main.py`

TODO: create Docker for easier setup


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
