import io

import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Skin acne Detector", layout="centered")

# Load the model once and cache it so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return YOLO("acne.pt")

model = load_model()

st.title("Skin Acne Detector")
st.write("Upload a image to detect acne.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    with st.spinner("Running detection..."):
        results = model(image)
        result = results[0]

    # Collect detected labels + confidence scores
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])
        detections.append((label, confidence))

    # Get the image with boxes drawn on it
    annotated = result.plot()  # numpy array in BGR
    annotated_image = Image.fromarray(annotated[:, :, ::-1])  # convert BGR -> RGB

    st.image(annotated_image, caption="Detection result", use_container_width=True)

    if detections:
        st.subheader("Detections")
        for label, confidence in detections:
            st.write(f"**{label}** — {confidence * 100:.1f}% confidence")
    else:
        st.info("No acne detected.")
