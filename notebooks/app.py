import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit.web.server.server
import numpy as np
import json 
import os
import hashlib

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_classifier_model.keras')
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load the model: {e}")
    os.remove(model_path)  # Delete the corrupted file
    st.error("Model file is corrupted. Please restart the app to download again.")
    exit()


# Calculate and display file size
file_size = os.path.getsize(model_path)
st.write("Deployed model file size (bytes):", file_size)

# Calculate and display MD5 checksum
def calculate_md5(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

md5_checksum = calculate_md5(model_path)
st.write("Deployed model MD5 checksum:", md5_checksum)

# Load class labels from JSON file
class_labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_labels.json')
with open(class_labels_path, 'r') as f:
    class_labels = json.load(f)


# Streamlit UI
st.title("Food Image Classifier")
st.write("Upload an image of food, and the model will predict its category!")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp.jpg')
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image
    img = load_img(temp_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[str(predicted_class)]
    confidence = predictions[0][predicted_class]

    # Set a confidence threshold
    confidence_threshold = 0.8
    if confidence < confidence_threshold:
        st.write("### This image does not resemble any of the following food categories:")
        for category in class_labels.values():
            st.write(f"- {category}")
    else:
        predicted_label = class_labels[str(predicted_class)]
        # Display results
        st.image(temp_path, caption="Uploaded Image", use_column_width=True)
        st.write(f"### Predicted Category: {predicted_label}")
        st.write(f"### Prediction Confidence: {confidence:.2f}")


