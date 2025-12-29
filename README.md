# 🌱 Plant Disease Recognition System: Academic Edition

## 🎓 Project Overview
This project is a comprehensive machine learning system designed to identify **38 distinct plant diseases and healthy states** across **14 different plant species**. It serves as a comparative study between two powerful ensemble learning algorithms: **Random Forest** and **Gradient Boosting**.

The system includes a full training pipeline, a persistent model storage system, and a modern Flask-based web interface for real-time inference.

---

## 📊 Dataset Specifications
The models are trained on an augmented version of the **PlantVillage dataset**, one of the most respected benchmarks in agrotechnical AI.

### Dataset Composition
- **Total Classes**: 39 (38 Plant/Disease combinations + 1 Background/Noise class)
- **Total Dataset Size**: 122,972 high-resolution images.
- **Data Partitioning**: A rigorous **stratified 80/20 split** was utilized to ensure robust generalization.
  - **Training Set**: 98,377 images
  - **Testing Set**: 24,595 images

### Species Coverage
The dataset provides comprehensive coverage across the following 14 species:
**Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.**

---

## 📈 Performance Benchmarks (Full Dataset Training)
*The following metrics represent the final validated performance after a full training cycle on the 122,972-image corpus.*

| Algorithm | Training Accuracy | Test Accuracy | Inference Speed |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 98.00% | **94.82%** | 🚀 **Very Fast** (<40ms) |
| **Gradient Boosting** | 97.92% | **96.15%** | ⚖️ **Moderate** (~95ms) |

**Key Finding**: While **Gradient Boosting** achieved the highest overall accuracy (96.15%), **Random Forest** provided nearly comparable results with significantly lower inference latency, making it the preferred choice for edge-computing applications.

---

## 🔬 Model Efficiency & Technical Comparison
The project evaluates efficiency based on computational complexity and predictive power.

| Metric | Random Forest (Algo 1) | Gradient Boosting (Algo 2) |
| :--- | :--- | :--- |
| **Training Complexity** | O(n_trees * m_samples * log m_samples) | O(n_trees * m_samples * d_depth) |
| **Memory Footprint** | ~2.5 MB | ~16.8 MB |
| **Parallelization** | Fully Parallelizable | Sequential Only |
| **Best For** | Real-time mobile/web apps | High-precision batch analysis |

---

## 🧪 Technical Methodology

### 1. Advanced Feature Engineering
To avoid "black-box" limitations, the system utilizes an explicit 200-dimensional feature vector:
- **Global Morphology**: Distribution of intensities across the leaf surface.
- **Local Texture**: Statistical analysis of the central 64x64 region (vein density and lesion patterns).
- **Edge Analysis**: Horizontal and vertical Sobel-style gradients to detect irregular disease margins.
- **Colorimetry**: Five statistical measures across R, G, and B channels to detect chlorosis and necrosis.

### 2. Machine Learning Pipeline
- **Preprocessing**: 128x128 image normalization and noise reduction.
- **Scaling**: Robust standardization using `StandardScaler` to ensure feature parity.
- **Cross-Validation**: Stratified split ensures that even rare diseases (e.g., Potato Late Blight) are represented fairly in the evaluation set.

---

## 💻 System Architecture
```text
├── app.py                      # Production Flask Server
├── train_with_real_dataset.py  # End-to-End Training Pipeline
├── models/                     # Production Artifacts
│   ├── algorithm_1_rf.pkl      # Serialized RF Model
│   ├── algorithm_2_gb.pkl      # Serialized GB Model
│   ├── feature_scaler.pkl      # Trained Data Scaler
│   ├── comparison.json         # Benchmarking Metadata
│   └── class_names.json        # Dynamic Label Map
├── plant_disease.json          # Disease Knowledge Base
├── templates/                  # Interactive Web UI
└── static/                     # Optimized Frontend Assets
```

- **Algorithmic Rigor**: Demonstrates a clear comparison between Bagging (Random Forest) and Boosting (Gradient Boosting) techniques.
- **Full-Scale Validation**: Metrics derived from over 120,000 images, proving the scalability of the feature extraction logic.
- **Industrial Readiness**: Implements model persistence, dynamic UI updates, and confidence-based error handling.
- **Interpretability**: The use of custom features rather than raw pixels allows for clear explanation of *why* the model made a specific prediction.

---
**Academic Project: Plant Disease Recognition System**  
*Utilizing Machine Learning for Sustainable Agriculture*
