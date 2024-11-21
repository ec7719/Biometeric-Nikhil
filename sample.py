import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Parameters
IMG_SIZE = (128, 128)
DATA_DIR_TRAIN = "EarTrainAMI/"
DATA_DIR_TEST = "EarTestAMI/"

# Function to load images and labels
def load_images(data_dir):
    images, labels = [], []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img)
                    labels.append(class_dir)
    return np.array(images), np.array(labels)

# Load training data
X_train, y_train = load_images(DATA_DIR_TRAIN)
X_test, y_test = load_images(DATA_DIR_TEST)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = to_categorical(label_encoder.fit_transform(y_train))
y_test_encoded = to_categorical(label_encoder.transform(y_test))

# Split training data into train and validation sets
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_train_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_val, y_val_encoded),
    epochs=50,
    batch_size=32
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_acc}")

# Function to predict a single image
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_class[0]
    return "Invalid image!"

# Test with a sample image
sample_image = "ear.jpg"
predicted_user = predict_image(sample_image)
print(f"Predicted User: {predicted_user}")
