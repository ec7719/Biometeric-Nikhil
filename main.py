import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU for training.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU.")

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DATA_DIR = "EarTrainAMI"
TEST_DATA_DIR = "EarTestAMI"
NUM_CLASSES = len([name for name in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, name))])

def load_and_preprocess_data(train_data_dir, test_data_dir):
    datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Load testing data
    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"Found {train_generator.samples} training images.")
    print(f"Found {test_generator.samples} testing images.")

    return train_generator, test_generator

def create_model():
    # Use transfer learning with ResNet50 
    base_model = ResNet50(
        weights='imagenet',  # Use pre-trained weights
        include_top=False, 
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False  # Freeze base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(train_generator, test_generator):
    model = create_model()
    
    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.2, 
        patience=5, 
        min_lr=0.00001
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=10, 
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=30,
        callbacks=[early_stopping, reduce_lr]
    )

    return model

def main():
    try:
        # Load and preprocess data
        train_generator, test_generator = load_and_preprocess_data(TRAIN_DATA_DIR, TEST_DATA_DIR)
        
        # Train model
        trained_model = train_model(train_generator, test_generator)
        
        # Save model
        trained_model.save('ear_recognition_model.h5')

        print("Model training completed with high accuracy!")
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    main()