import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Parameters
IMG_SIZE = (128, 128)
MODEL_PATH = "ear_classifier_model.h5"  # Path to your trained model
LABELS = ['class1', 'class2']  # Update with actual class names

# Load the trained model
model = load_model(MODEL_PATH)

# Pretrained label encoder (assumes same encoder used during training)
label_encoder = LabelEncoder()
label_encoder.fit(LABELS)

# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the user from the frame
def predict_user(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 's' to capture and predict, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Display the webcam feed
    cv2.imshow("Webcam - Ear Recognition", frame)

    # Key press actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to capture and predict
        predicted_user = predict_user(frame)
        print(f"Predicted User: {predicted_user}")
        # Overlay the prediction on the frame
        cv2.putText(frame, f"User: {predicted_user}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", frame)
        cv2.waitKey(2000)  # Display the prediction for 2 seconds

    elif key == ord('q'):  # Press 'q' to quit
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
