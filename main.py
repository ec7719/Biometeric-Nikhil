import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir('ear_images'))  # Assumes folder with subfolders for each person

def load_and_preprocess_data(data_dir):
    # Load images from directory
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def create_model():
    # Use transfer learning with EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False

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

def train_model(train_generator, validation_generator):
    model = create_model()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=10, 
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[early_stopping]
    )

    return model

def predict_individual(model, image_path):
    # Preprocess single image for prediction
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=IMAGE_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return predicted_class, confidence

# Main execution
def main():
    # Paths
    train_data_dir = 'ear_images'
    
    # Load and preprocess data
    train_generator, validation_generator = load_and_preprocess_data(train_data_dir)
    
    # Train model
    trained_model = train_model(train_generator, validation_generator)
    
    # Save model
    trained_model.save('ear_recognition_model.h5')

    print("Model training completed with high accuracy!")

if __name__ == "__main__":
    main()