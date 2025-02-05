import cv2
import numpy as np
import tensorflow as tf
import yaml

# Load configuration settings
with open('../config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Load the pre-trained abnormal behavior detection model
abnormal_model = tf.keras.models.load_model(config['model_paths']['abnormal_model'])

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    """
    Preprocess the frame before feeding it to the abnormal behavior detection model.
    Resizes the frame and normalizes pixel values.

    Args:
        frame (numpy array): A single frame from the video feed.

    Returns:
        numpy array: Preprocessed frame ready for model input.
    """
    # Assuming the model expects input size of 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    # Add a batch dimension to the frame
    return np.expand_dims(normalized_frame, axis=0)

# Function to detect abnormal behavior in the given frame
def detect_abnormal_behavior(frame):
    """
    Predict whether the given frame shows abnormal behavior.

    Args:
        frame (numpy array): A single frame from the video feed.

    Returns:
        bool: True if abnormal behavior is detected, False otherwise.
    """
    preprocessed_frame = preprocess_frame(frame)
    
    # Make a prediction using the model
    prediction = abnormal_model.predict(preprocessed_frame)
    
    # Assume the model outputs a probability score (0 - normal, 1 - abnormal)
    abnormal_score = prediction[0][0]  # Get the abnormality score
    
    # Debugging: print the abnormal score
    print(f"Abnormal Score: {abnormal_score}")
    
    # Check if the score crosses a certain threshold
    threshold = config.get('abnormal_threshold', 0.8)  # Default threshold if not in config
    is_abnormal = abnormal_score >= threshold
    
    # Debugging: print whether the behavior is abnormal
    print(f"Is Abnormal: {is_abnormal}")
    
    return is_abnormal

# Testing the function with video feed (Optional)
if __name__ == "__main__":
    # Open video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Detect abnormal behavior
        if detect_abnormal_behavior(frame):
            cv2.putText(frame, "Abnormal Behavior Detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Abnormal Behavior Detection', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()