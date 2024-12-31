from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

# Load the saved model
model = load_model('food_classifier_model.keras')

# Load class labels from JSON file
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Debugging: Print loaded class labels
print("Loaded Class Labels:", class_labels)

# Load and preprocess an image
img = load_img('../data/test/pizza/1571074.jpg', target_size=(224, 224))  # Replace with your test image path
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Get the predicted class name
predicted_label = class_labels[str(predicted_class)]

# Print results
print(f"Predicted Class Index: {predicted_class}")
print(f"Predicted Class Name: {class_labels[predicted_class]}")
