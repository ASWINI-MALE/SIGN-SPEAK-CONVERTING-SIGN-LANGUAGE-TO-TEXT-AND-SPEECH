import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# Path to dataset 
DATASET_PATH = r"C:\Users\malea\Downloads\SignSpeak -Converting Sign Language to Text and Speech\SignSpeak -Converting Sign Language to Text and Speech"
# Image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Data Augmentation & Data Generator
datagen = ImageDataGenerator(
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Get class labels
GESTURE_LABELS = list(train_generator.class_indices.keys())

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(GESTURE_LABELS), activation='softmax')  # Dynamically set class count
])

# Compile Model
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy"])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model
EPOCHS = 30
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[early_stopping])

# Save the Model
model.save("sign_language_model.h5")
print("Model saved as sign_language_model.h5")
 