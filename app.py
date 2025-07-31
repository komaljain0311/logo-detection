import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# Load the trained YOLOv8 model
model = YOLO(r'G:\logo\runs\detect\train\weights\best.pt')  # adjust if needed

st.set_page_config(page_title="Logo Detection App", layout="centered")
st.title("🔍 Logo Detection with YOLOv8")
st.markdown("Upload an image or video to detect logos using your custom-trained model.")

option = st.radio("Choose Input Type", ['Image', 'Video'])

if option == 'Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Logos"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                img.save(temp.name)
                results = model(temp.name)

            for r in results:
                result_img = r.plot()  # Annotated image
                st.image(result_img, caption="Detected Logos", use_column_width=True)

elif option == 'Video':
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.button("Detect Logos in Video"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR")

            cap.release()