# 🤟 BSL Hand Gesture Recognition System

A deep learning system that classifies **British Sign Language (BSL)** hand gestures with **95.74% accuracy** across 34 classes — numbers (0–10) and alphabet (excluding H, J, Y).

🌐 **Live Demo:** [bsl-handgesturerecognition.streamlit.app](https://bsl-handgesturerecognition-rhu9sdhz9v9pzvxufdw2w7.streamlit.app)

---

## 📌 Overview

| Detail | Info |
|---|---|
| Dataset | 34,000 images · 34 classes · 1,000 per class |
| Model | MobileNetV2 (Transfer Learning + Fine-tuning) |
| Accuracy | 95.74% |
| F1-Score | 95.78% |
| Deployment | Streamlit Web App |

---

## 🗂️ Dataset

**BSL Hand Gesture Dataset** by Eren Tatepe  
📦 [Kaggle Dataset](https://www.kaggle.com/datasets/drmabroukaabuhmida/bsl-dataset-credits-to-eren-tatepe)

- 34,000 images across 34 BSL gesture classes
- Numbers: Zero (0) to Ten (10)
- Alphabet: A–Z excluding H, J, Y
- Perfectly balanced — 1,000 images per class

---

## 🔧 Pipeline

```
Raw Image
    │
    ▼
CLAHE Normalisation (LAB L-channel, clipLimit=2.0)
    │
    ▼
MediaPipe Hand Landmarker (21 keypoints)
    │
    ▼
Bounding Box + 20% Padding → Crop
    │
    ▼
Resize to 224×224 → Normalise [0,1]
    │
    ▼
MobileNetV2 → Softmax (34 classes)
```

---

## 🧠 Model Architecture

- **Base:** MobileNetV2 pre-trained on ImageNet (frozen in Phase 1)
- **Head:** GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(34, Softmax)
- **Phase 1:** Train head only · Adam lr=1e-3 · 10 epochs · Best val acc: 82.87%
- **Phase 2:** Unfreeze last 30 layers · Adam lr=1e-4 · 20 epochs · Best val acc: **95.74%**

---

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | 95.74% |
| Precision | 96.04% |
| Recall | 95.74% |
| F1-Score | 95.78% |
| Classes ≥ 90% Recall | 30 / 34 |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yoursmaddyy/bsl-gesture-recognition.git
cd bsl-gesture-recognition
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add model files**

Download and place in the root directory:
- `mobilenetv2_bsl_final.h5` — trained model weights
- `class_indices.json` — class index mapping

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
bsl-gesture-recognition/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── class_indices.json      # Class index mapping
├── hand_landmarker.task    # MediaPipe model (auto-downloaded)
├── notebook.ipynb          # Training notebook
└── README.md
```

---

## 📦 Requirements

```
streamlit
tensorflow
opencv-python
mediapipe
numpy
Pillow
scikit-learn
```

---

## ⚠️ Notes

- MediaPipe `hand_landmarker.task` is automatically downloaded at first run.
- GPU recommended for training (Tesla T4 used via Kaggle).

---

## 🙏 Credits

- Dataset: [Eren Tatepe via Kaggle](https://www.kaggle.com/datasets/drmabroukaabuhmida/bsl-dataset-credits-to-eren-tatepe)
- MediaPipe: [Google](https://mediapipe.dev)
- MobileNetV2: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)

---

