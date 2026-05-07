import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

# Page Config
st.set_page_config(
    page_title="AI Image Authenticator",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 AI Powered Face Image Authenticator")
st.markdown(
    "Upload an image to detect whether it is **Real** or **AI Generated**."
)

# Load Model
@st.cache_resource
def load_model():
    try:
        # Make sure your model file exists
        model = joblib.load("model.jbl")
        return model

    except FileNotFoundError:
        st.error("⚠️ model.jbl not found.")
        return None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# Upload Image
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file).convert("RGB")

    # Show Image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Analyze Button
    analyze = st.button("Analyze Image")

    if analyze:

        if model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Analyzing Image..."):

                # Convert image to numpy array
                img_array = np.array(image)

                # Resize image
                img_resized = cv2.resize(img_array, (128, 128))

                # Normalize
                img_norm = img_resized / 255.0

                # Reshape for model
                img_input = np.expand_dims(img_norm, axis=0)

                # Prediction
                prediction = model.predict(img_input)

                # Convert prediction safely
                score = float(prediction[0][0])

                # Classification
                if score > 0.5:
                    label = "🤖 AI Generated"
                    confidence = score * 100
                    color = "red"
                else:
                    label = "✅ Real Image"
                    confidence = (1 - score) * 100
                    color = "green"

            # Result
            st.markdown(
                f"""
                <h2 style='text-align:center; color:{color};'>
                    {label}
                </h2>
                """,
                unsafe_allow_html=True
            )

            st.success(f"Confidence: {confidence:.2f}%")

            # Progress Bar
            st.progress(min(max(score, 0.0), 1.0))

            # Image Info
            width, height = image.size

            st.info(
                f"""
                File Name: {uploaded_file.name}

                Resolution: {width} x {height}

                File Size: {uploaded_file.size / 1024:.2f} KB
                """
            )