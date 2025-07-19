# MultiDog-BreedClassifier

This project detects **multiple dogs** in a single image, crops each, and predicts their **breeds** using a fine-tuned ResNet50 model. It uses YOLOv8 for object detection and ResNet50 for image classification.

---

## 🔗 Live Demo

**[Click here to try the app ](https://huggingface.co/spaces/Razor2507/DogsBreedClassification)**  

---

## Features

- Detects **multiple dogs** in one image (YOLOv8)
- Crops and processes each dog individually
- Predicts breed with a **ResNet50 CNN classifier**
- Full-stack deployment with **React (Frontend)** + **FastAPI (Backend)**
- Upload image → See labeled results

---

## Model Overview

| Component | Details |
|----------|---------|
| **Object Detector** | YOLOv8 (pretrained) |
| **Classifier** | ResNet50 (fine-tuned on 120 breeds) |
| **Framework** | PyTorch |
| **Deployment** | React + FastAPI |

---

## Project Structure
📁 model/            → Trained ResNet50 model and breed classes (.pth)  
📁 routes/           → FastAPI route definitions (prediction logic)  
📁 notebook/         → Jupyter notebooks for training and testing  
📁 dist/             → Frontend build folder generated from React which contains GUI.   
📄 yolov8n.pt        → YOLOv8n pretrained weights for dog detection  
📄 classes.json      → Mapping of class indices to dog breed names  
📄 main.py           → Entry point for FastAPI app  
📄 requirements.txt  → Python dependencies  
📄 readme.md         → Project documentation (this file)  


## Installation

```bash
# Clone the repo
git clone https://github.com/vedant250702/MultiDog-BreedClassifier.git
cd dog-breed-classification

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
python main.py
