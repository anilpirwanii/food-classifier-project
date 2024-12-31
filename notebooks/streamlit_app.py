import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit.web.server.server
import numpy as np
import json 
import os

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_classifier_model.keras')
model = load_model(model_path)

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

    # # Debugging: Print the loaded class labels
    # st.write("Loaded Class Labels:", class_labels)

    # # Debugging: Print the model's predictions
    # predictions = model.predict(img_array)
    # st.write("Raw Predictions (Probabilities):", predictions)

    # # Debugging: Print the predicted class index
    # predicted_class = np.argmax(predictions)
    # st.write("Predicted Class Index:", predicted_class)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[str(predicted_class)]

    # Display results
    st.image(temp_path, caption=f"Uploaded Image", use_column_width=True)
    st.write(f"### Predicted Category: {predicted_label}")
    st.write(f"### Prediction Confidence: {predictions[0][predicted_class]:.2f}")
