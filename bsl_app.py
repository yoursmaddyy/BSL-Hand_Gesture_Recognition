import streamlit as st
import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from PIL import Image
import urllib.request
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BSL Hand Gesture Classifier",
    page_icon="🤟",
    layout="centered"
)

# ── Load Model & Classes ──────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model('mobilenetv2_bsl_final.h5')

    with open('class_indices.json', 'r') as f:
        idx_to_class = json.load(f)

    # Download MediaPipe model if not present
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            model_path
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    return model, idx_to_class, detector

model, idx_to_class, detector = load_resources()

# ── Helper Functions ──────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(image_rgb, landmarks):
    h, w, _ = image_rgb.shape
    annotated = image_rgb.copy()
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(annotated, pts[start], pts[end], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(annotated, pt, 5, (255, 0, 0), -1)
        cv2.circle(annotated, pt, 5, (255, 255, 255), 1)
    return annotated

def get_bounding_box(landmarks, image_shape, padding=0.2):
    h, w, _ = image_shape
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    x_min_px = int(min(x_coords) * w)
    x_max_px = int(max(x_coords) * w)
    y_min_px = int(min(y_coords) * h)
    y_max_px = int(max(y_coords) * h)
    pad_x = int((x_max_px - x_min_px) * padding)
    pad_y = int((y_max_px - y_min_px) * padding)
    return (max(0, x_min_px - pad_x), max(0, y_min_px - pad_y),
            min(w, x_max_px + pad_x), min(h, y_max_px + pad_y))

def apply_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)

def predict(image_bgr):
    # Step 1: CLAHE
    enhanced = apply_clahe(image_bgr)
    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    # Step 2: MediaPipe detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results  = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None, None, None, None

    # Step 3: Crop hand region
    landmarks = results.hand_landmarks[0]
    x1, y1, x2, y2 = get_bounding_box(landmarks, enhanced.shape)
    cropped = enhanced[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None, None, None

    # Step 4: Preprocess for model
    resized    = cv2.resize(cropped, (224, 224))
    normalized = resized / 255.0
    input_arr  = np.expand_dims(normalized, axis=0)

    # Step 5: Predict
    preds      = model.predict(input_arr, verbose=0)
    top3_idx   = np.argsort(preds[0])[::-1][:3]
    top3       = [(idx_to_class[str(i)], float(preds[0][i]) * 100) for i in top3_idx]
    pred_class = top3[0][0]
    confidence = top3[0][1]

    # Step 6: Draw landmarks on original
    annotated = draw_landmarks(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), landmarks)

    return pred_class, confidence, top3, annotated

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤟 BSL Hand Gesture Classifier")
st.markdown("Upload an image of a **British Sign Language (BSL)** hand gesture to classify it.")
st.markdown("---")

tab1, tab2 = st.tabs(["📁 Upload Image", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Show uploaded image
        st.subheader("Uploaded Image")
        st.image(image_rgb, use_column_width=True)

        # Predict
        with st.spinner("Detecting hand and classifying..."):
            pred_class, confidence, top3, annotated = predict(image_bgr)

        st.markdown("---")

        if pred_class is None:
            st.error("❌ No hand detected in this image. Please try another image.")
        else:
            # Results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖐️ Detected Landmarks")
                st.image(annotated, use_column_width=True)

            with col2:
                st.subheader("🎯 Prediction")
                st.markdown(f"### `{pred_class}`")
                st.metric("Confidence", f"{confidence:.2f}%")

                st.subheader("Top 3 Predictions")
                for rank, (cls, conf) in enumerate(top3, 1):
                    st.progress(int(conf), text=f"{rank}. {cls} — {conf:.2f}%")

with tab2:
    st.subheader("About This App")
    st.markdown("""
    This application classifies **British Sign Language (BSL)** hand gestures using a deep learning pipeline:

    **Pipeline:**
    1. 📸 Image uploaded by user
    2. 💡 CLAHE applied for lighting normalization
    3. 🖐️ MediaPipe detects and crops the hand region
    4. 🧠 MobileNetV2 classifies the gesture
    5. 📊 Top 3 predictions shown with confidence scores

    **Model:** MobileNetV2 (Transfer Learning + Fine-tuning)  
    **Dataset:** BSL Hand Gesture Dataset  
    **Preprocessing:** MediaPipe Landmark Detection + CLAHE
    """)