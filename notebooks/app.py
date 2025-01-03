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
import warnings

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings('ignore')

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_classifier_model.keras')
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    st.error("Failed to load the model. Please check the model file and try again.")
    st.stop()

# Load class labels from JSON file
class_labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_labels.json')
try:
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)
except Exception as e:
    st.error("Failed to load class labels. Please check the JSON file.")
    st.stop()


# Streamlit UI
st.title("Food Image Classifier")
st.write("Upload an image of food, and the model will predict its category!")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp.jpg')
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error("Failed to save the uploaded file. Please try again.")
        st.stop()

    # Preprocess the image
    img = load_img(temp_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        predicted_label = class_labels[str(predicted_class)]
    except Exception as e:
        st.error("Prediction failed. Please check the model or input image.")
        st.stop()

    # Set a confidence threshold
    confidence_threshold = 0.8
    if confidence < confidence_threshold:
        st.write("### This image does not resemble any of the following food categories:")
        for category in class_labels.values():
            st.write(f"- {category}")
    else:
        predicted_label = class_labels[str(predicted_class)]
        # Display results
        st.image(temp_path, caption="Uploaded Image", use_container_width=True)
        st.write(f"### Predicted Category: {predicted_label}")
        st.write(f"### Prediction Confidence: {confidence:.2f}")

    # Clean up temporary file
    try:
        os.remove(temp_path)
    except Exception as e:
        pass


