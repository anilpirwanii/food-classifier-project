import tensorflow as tf
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to train and test directories
train_dir = '../../food-101/train'
test_dir = '../../food-101/test'

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Shift the image width-wise up to 20%
    height_shift_range=0.2, # Shift the image height-wise up to 20%
    shear_range=0.2,        # Shear the image
    zoom_range=0.2,         # Zoom in/out
    horizontal_flip=True,   # Flip the image horizontally
    brightness_range=[0.8, 1.2],  # Adjust brightness
    fill_mode='nearest'     # Fill missing pixels after transformations
)

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

# Get the class labels
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Save the class labels as a JSON file
with open("class_labels.json", "w") as f:
    json.dump(class_labels, f)

print("Class labels saved to class_labels.json")

# Unfreeze specific layers of MobileNetV2 for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:150]:  # Freeze the first 150 layers
    layer.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax') 
])

# Compile with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Print a summary of the model

print("Train Classes:", train_generator.class_indices)
print("Number of Train Classes:", len(train_generator.class_indices))
print("Test Classes:", test_generator.class_indices)
print("Number of Test Classes:", len(test_generator.class_indices))


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_food_classifier_model.keras', 
    save_best_only=True, 
    monitor='val_loss'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=2, 
    min_lr=1e-6
)

# Increase epochs and use callbacks
history = model.fit(
    train_generator,
    epochs=10,  # Fewer epochs
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)


model.save('food_classifier_model.keras')

