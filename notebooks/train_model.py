import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to train and test directories
train_dir = '../data/train'
test_dir = '../data/test'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the MobileNetV2 model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),  # Input size for MobileNetV2
    include_top=False,         # Exclude the top (classification) layer
    weights='imagenet'         # Pre-trained on ImageNet
)

# Freeze the base model layers (no training for these layers)
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax') 
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Print a summary of the model

print("Train Classes:", train_generator.class_indices)
print("Number of Train Classes:", len(train_generator.class_indices))
print("Test Classes:", test_generator.class_indices)
print("Number of Test Classes:", len(test_generator.class_indices))


history = model.fit(
    train_generator,
    epochs=5,  # Start with a small number; we can increase later
    validation_data=test_generator
)

model.save('food_classifier_model.keras')

