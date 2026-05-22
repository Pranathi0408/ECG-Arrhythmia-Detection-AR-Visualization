from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# Dataset folder
dataset_path = "Dataset"

# Count number of class folders
num_classes = len(os.listdir(dataset_path))
print("Detected Classes:", num_classes)

# Image settings
img_size = (48, 48)
batch_size = 16

# Data Generator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Save class labels
labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}   # Reverse mapping
os.makedirs("Model", exist_ok=True)
np.save("Model/labels.npy", labels)
print("Class Labels Saved:", labels)

# Build CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# Compile
model.compile(optimizer=Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(train_generator,
          validation_data=val_generator,
          epochs=20)

# Save Model
model.save("Model/model.h5")
print("✅ Training Completed — Model Saved in /Model folder")
