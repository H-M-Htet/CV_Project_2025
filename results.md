# Helmet Violation Detection System (YOLOv8)

This project implements an automated helmet-violation detection system using a custom-trained **YOLOv8-s** model.  
The system identifies motorcycle riders, determines whether they are wearing a helmet, and associates each rider with the nearest license plate for further processing.  
This work was developed as part of the Computer Vision 2025 class project.

---

## 🚀 Features

### ✔ Helmet Detection (3 Classes)
The YOLOv8-s detector was trained to classify:

- **0 — No_helmet**
- **1 — Wearing_helmet**
- **2 — Person_on_Bike**

### ✔ Violation Detection
A custom association module flags violations when:
- The rider is classified as **No_helmet**, or  
- A **Person_on_Bike** does not overlap with any **Wearing_helmet** box (IoU < 0.3)

### ✔ Plate Detection (Prototype)
A lightweight YOLO model is used to detect license plates for later OCR processing.

### ✔ Rider–Plate Association
The system matches each violating rider with the **nearest plate** based on Euclidean distance.

---

## 📦 Dataset & Training

### Dataset Preparation
- Multiple datasets merged into a single 3-class format  
- Oversampling applied to **No_helmet** class  
- Data split into **70% train**, **20% validation**, **10% test**

### Model
- Model: **YOLOv8-s**
- Image size: **640**
- Epochs: **100**
- Augmentation: blur, brightness/contrast, perspective, coarse dropout
- Framework: **Ultralytics YOLO**

---

## 📊 Model Performance

### Validation Metrics
- **Precision:** 0.975  
- **Recall:** 0.975  
- **mAP@0.5:** 0.986  
- **mAP@0.5:0.95:** 0.768  

### Per-Class AP@0.5
| Class            | AP50 |
|------------------|------|
| No_helmet        | 0.989 |
| Wearing_helmet   | 0.977 |
| Person_on_Bike   | 0.993 |

### Test Set (10%)
Results closely match validation performance, showing strong generalization.

---

## 🧩 Association Module

The association module maps each detected rider to:

1. Helmet class  
2. Motorcycle  
3. Nearest license plate (based on centroid distance)  

A violation is flagged when:
- **Class 0 = No_helmet**, or  
- **Class 2 = Person_on_Bike** and **no overlapping helmet** (IoU < 0.3)

**Flowchart available in the repo** for detailed visualization.

---

## 🔍 Sample Outputs
- Helmet/no-helmet detection  
- Motorcycle rider detection  
- Violation visualization  
- Plate detection samples  

*(Add images from `runs/detect/val*/` here)*

---

## ⚠ OCR Status

OCR integration was attempted but **not completed** due to issues with:
- low plate resolution  
- motion blur  
- harsh lighting  
- partial occlusions  

OCR integration is listed as **future work**.

---

## 🛑 Limitations

- OCR incomplete  
- Plate detection accuracy lower than helmet detection  
- No temporal tracking (single-frame analysis)  
- Real-time optimization not yet implemented

---

## 🔮 Future Work

- Improve plate dataset + apply super-resolution  
- Integrate OCR (Thai + English) end-to-end  
- Add tracking (ByteTrack / DeepSORT)  
- Test in real traffic videos  
- Deploy on Jetson / edge hardware

---

## 📁 Project Structure

