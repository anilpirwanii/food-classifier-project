from tensorflow.keras.models import load_model

# Load the existing .h5 model
model = load_model('food_classifier_model.h5')

# Save it in the recommended .keras format
model.save('food_classifier_model.keras')
print("Model saved in .keras format!")
