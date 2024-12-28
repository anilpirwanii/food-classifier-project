from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = load_model('food_classifier_model.keras')

# Load and preprocess an image
img = load_img('../data/test/pizza/1571074.jpg', target_size=(224, 224))  # Replace with your test image path
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Mapping class index to class name
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use a dummy generator to get class_indices (matches the one in train_model.py)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '../data/train', target_size=(224, 224), batch_size=32, class_mode='categorical'
)
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Print results
print(f"Predicted Class Index: {predicted_class}")
print(f"Predicted Class Name: {class_labels[predicted_class]}")
