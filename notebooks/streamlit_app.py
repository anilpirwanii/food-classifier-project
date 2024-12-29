import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit.web.server.server
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_classifier_model.keras')
model = load_model(model_path)

# Get class labels
train_datagen = ImageDataGenerator(rescale=1./255)
train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/train')
train_generator = train_datagen.flow_from_directory(
    train_data_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

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
    predicted_label = class_labels[predicted_class]

    # Display results
    st.image(temp_path, caption=f"Uploaded Image", use_column_width=True)
    st.write(f"### Predicted Category: {predicted_label}")
    st.write(f"### Prediction Confidence: {predictions[0][predicted_class]:.2f}")
