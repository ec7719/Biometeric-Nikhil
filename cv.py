import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class EarRecognizer:
    def __init__(self, model_path, labels):
        # Load pre-trained model
        self.model = load_model(model_path)
        self.labels = labels
        self.image_size = (224, 224)

    def preprocess_frame(self, frame):
        # Resize and normalize frame for model input
        resized = cv2.resize(frame, self.image_size)
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor

    def recognize_ear(self, ear_image):
        # Preprocess and predict
        processed_image = self.preprocess_frame(ear_image)
        predictions = self.model.predict(processed_image)
        
        # Get top prediction
        top_prediction_index = np.argmax(predictions[0])
        confidence = predictions[0][top_prediction_index]
        label = self.labels[top_prediction_index]
        
        return label, confidence

    def real_time_recognition(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Recognize ear
                label, confidence = self.recognize_ear(frame)
                
                # Display results
                display_text = f"{label} (Confidence: {confidence:.2f})"
                cv2.putText(frame, 
                            display_text, 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 255, 0), 
                            2)
            except Exception as e:
                print(f"Recognition error: {e}")
            
            # Display the resulting frame
            cv2.imshow('Ear Recognition', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Load labels (assuming labels are folder names in your dataset)
    import os
    labels = os.listdir('ear_images')
    
    # Initialize recognizer
    ear_recognizer = EarRecognizer(
        model_path='ear_recognition_model.h5', 
        labels=labels
    )
    
    # Start real-time recognition
    ear_recognizer.real_time_recognition()

if __name__ == "__main__":
    main()