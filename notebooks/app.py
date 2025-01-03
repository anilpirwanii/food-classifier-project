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
# Add title and subtitle
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 3rem; font-weight: bold; margin-bottom: 10px; color: #333333;">
            Welcome to Snack Scanner
        </h1>
        <h2 style="font-size: 1.5rem; font-weight: normal; color: #666666;">
            Your AI-Powered Food Recognition Companion
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #333333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #666666;
            text-align: center;
            margin-bottom: 30px;
        }
        .contact-card {
            text-align: center;
            padding: 10px;
            font-size: 14px;
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            border-radius: 10px;
            border: 1px solid #f1f1f1;
            margin-top: 20px;
        }
        .contact-card a {
            text-decoration: none;
            color: #0073e6;
        }
        .contact-card a:hover {
            color: #005bb5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    confidence_threshold = 0.7
    # When the model cannot classify the image
    if confidence < confidence_threshold:
        st.markdown(
            """
            <div style="border: 1px solid #f1f1f1; border-radius: 10px; padding: 20px; background-color: #f9f9f9; text-align: center; font-family: Arial, sans-serif;">
                <h3 style="color: #FF4B4B;">❌ Sorry, we couldn't confidently classify this image!</h3>
                <p style="font-size: 16px; color: #333;">This image does not resemble any of the following food categories:</p>
                <ul style="list-style: none; padding: 0; text-align: left; display: inline-block;">
                    <li>🥗 Caesar Salad</li>
                    <li>🍗 Chicken Wings</li>
                    <li>🍰 Chocolate Cake</li>
                    <li>🐟 Fish and Chips</li>
                    <li>🍟 French Fries</li>
                    <li>🌭 Hot Dog</li>
                    <li>🍦 Ice Cream</li>
                    <li>🍕 Pizza</li>
                    <li>🍁🍟 Poutine</li>
                    <li>🍜 Ramen</li>
                    <li>🥟 Samosa</li>
                    <li>🥩 Steak</li>
                    <li>🍣 Sushi</li>
                    <li>🌮 Tacos</li>
                    <li>🧇 Waffles</li>
                </ul>
                <p style="font-size: 16px; color: #333; margin-top: 20px;">📸 To improve accuracy:</p>
                <ul style="list-style: disc; padding: 0 20px; text-align: left; display: inline-block;">
                    <li>Ensure the image contains only one food item.</li>
                    <li>Crop out the background to focus on the food.</li>
                    <li>Avoid including multiple food items in one image.</li>
                </ul>
                <p style="font-size: 16px; color: #333; margin-top: 20px;">🔄 Try uploading another image and give it another shot!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    emoji_dict = {
    "caesar_salad": "🥗",
    "chicken_wings": "🍗",
    "chocolate_cake": "🍰",
    "fish_and_chips": "🐟",
    "french_fries": "🍟",
    "hot_dog": "🌭",
    "ice_cream": "🍦",
    "pizza": "🍕",
    "poutine": "🍁🍟",
    "ramen": "🍜",
    "samosa": "🥟",
    "steak": "🥩",
    "sushi": "🍣",
    "tacos": "🌮",
    "waffles": "🧇",
}

    if confidence >= confidence_threshold:
        emoji = emoji_dict.get(predicted_label, "🍴")  # Default emoji if category not found
        st.markdown(
            f"""
            <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #eaeaea;">
                <h1 style="color: #4CAF50;">{emoji} {predicted_label.replace("_", " ").title()}</h1>
                <p style="font-size: 18px;">Prediction Accuracy: <b>{confidence * 100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "The model could not confidently classify this image as one of the categories. Try another image!"
        )

    # Clean up temporary file
    try:
        os.remove(temp_path)
    except Exception as e:
        pass

# Add contact information
st.markdown(
    """
    <hr style="border:1px solid var(--color-primary);">
    <div style="text-align: center; padding: 20px; font-size: 16px; font-family: Arial, sans-serif; background-color: var(--background-secondary); border-radius: 15px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); border: 1px solid var(--color-primary);">
        <p style="margin: 10px 0;">🌟 Powered by <a href="https://github.com/anilpirwanii/food-classifier-project" target="_blank" style="text-decoration: none; color: var(--color-primary);"><b>Machine Learning-Based Snack Scanner</b></a></p>
        <p style="margin: 10px 0;">🍴 Created by <b>Anil Kumar</b></p>
        <p style="margin: 10px 0;">📧 <a href="mailto:aka158@sfu.ca" style="text-decoration: none; color: var(--color-primary);">aka158@sfu.ca</a></p>
        <p style="margin: 10px 0;">🌐 
            <a href="https://github.com/anilpirwanii/" target="_blank" style="text-decoration: none; color: var(--color-primary);">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/github.svg" alt="GitHub" style="width:20px; height:20px; vertical-align: middle;"> GitHub
            </a> 
            &nbsp;|&nbsp; 
            <a href="https://linkedin.com/in/anilpirwanii/" target="_blank" style="text-decoration: none; color: var(--color-primary);">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" style="width:20px; height:20px; vertical-align: middle;"> LinkedIn
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)



