# AI-Powered-Face-image-Authentication-System

🔍 AI-Powered Face Image Authentication System
A deep learning-based web application that detects whether an image is Real or AI Generated using a Convolutional Neural Network (CNN) and a Streamlit interface.

📌 Overview
With the rise of AI-generated images (deepfakes, GAN-generated faces, etc.), it has become increasingly difficult to distinguish real photos from fake ones. This project addresses that problem by training a CNN model on real and AI-generated face images and deploying it as an interactive web app.

🚀 Features

Upload any image (JPG, JPEG, PNG, WEBP)
Detects if the image is Real or AI Generated
Shows confidence score and AI probability score
Displays image metadata (filename, dimensions, file size)
Fast and lightweight CNN model
Clean and simple Streamlit UI


🧠 Model Architecture
The CNN model is built using TensorFlow/Keras:
Input (128x128x3)
→ Conv2D(32, 3x3, relu) → MaxPool2D(2x2)
→ Conv2D(64, 3x3, relu) → MaxPool2D(2x2)
→ Flatten
→ Dense(128, relu)
→ Dense(1, sigmoid)

Loss: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
Early Stopping: patience=10, monitors val_loss

🗂️ Dataset
The model was trained on 6000 images (3000 Real + 3000 AI Generated).

Real images: Genuine face photographs
Fake images: AI-generated face images
Split: 80% training / 20% testing
Image size: 128x128 pixels

🛠️ Tech Stack

Python
TensorFlow / Keras — Model training
OpenCV — Image processing
MediaPipe — Face detection
Scikit-learn — Data splitting & shuffling
Streamlit — Web interface
Joblib — Model saving/loading
