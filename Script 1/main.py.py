import threading
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from tensorflow.keras.preprocessing import image
from alert import send_email_alert
from utils import save_detected_image
import os

# Load pre-trained models for mask and weapon detection
mask_model = load_model(r'C:\Users\PranithaReddy\Desktop\mini project\theft detection\script\model\mask_detector.h5')
weapon_model = tf.keras.models.load_model(r'C:\Users\PranithaReddy\Desktop\mini project\theft detection\script\model\weapon_detector.h5')

# Use face detection instead of full-body detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Initialize the background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Motion detection sensitivity
motion_threshold = 155000  # Adjust based on environment

# Minimum confidence threshold for weapon detection
weapon_threshold = 0.99  # Adjust threshold if needed

# Create a directory to save cropped faces for debugging
cropped_faces_dir = 'cropped_faces_debug'
os.makedirs(cropped_faces_dir, exist_ok=True)

# Debounce settings
alert_sent = False
last_alert_time = 0
alert_interval = 10  # Seconds between alerts to avoid multiple triggers

# Abnormal behavior tracking
previous_position = None
abnormal_movement_threshold = 20  # Pixels for abnormal movement

# Function to save cropped faces for debugging
def save_cropped_face(face, count):
    save_path = os.path.join(cropped_faces_dir, f'cropped_face_{count}.jpg')
    cv2.imwrite(save_path, face)
    print(f"[DEBUG] Cropped face saved to: {save_path}")

# Function for mask detection
def detect_mask(face):
    try:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (64, 64))
        face_img = np.expand_dims(face_resized, axis=0) / 255.0
        prediction = mask_model.predict(face_img)
        predicted_value = prediction[0][0]
        print(f"[DEBUG] Mask Prediction Value: {predicted_value:.4f}")
        mask_threshold = 1.0
        return predicted_value > mask_threshold
    except Exception as e:
        print(f"[ERROR] Mask detection failed: {e}")
        return False

# Function for weapon detection
def detect_weapon(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.expand_dims(frame_resized, axis=0) / 255.0
    prediction = weapon_model.predict(frame_resized)
    confidence = prediction[0]
    print(f"[DEBUG] Weapon Prediction Confidence: {confidence[0]:.4f}")
    return confidence, confidence[0] > weapon_threshold

# Function for motion detection
def detect_motion(fgbg, frame):
    fgmask = fgbg.apply(frame)
    motion_value = np.sum(fgmask)
    fgmask[fgmask < 127] = 0
    fgmask[fgmask >= 127] = 255
    return motion_value > motion_threshold

# Function to save detected image in a separate thread
def threaded_save_image(frame):
    threading.Thread(target=save_detected_image, args=(frame,)).start()

# Function to send an email alert in a separate thread
def threaded_send_alert(image_path):
    threading.Thread(target=send_email_alert, args=(image_path,)).start()

# Function to check for abnormal behavior based on movement
def detect_abnormal_behavior(faces):
    global previous_position
    if len(faces) == 0:
        return False

    # Get the centroid of the first detected face
    x, y, w, h = faces[0]
    centroid = (x + w // 2, y + h // 2)

    if previous_position is not None:
        # Calculate the distance between the current and previous centroid positions
        distance = np.sqrt((centroid[0] - previous_position[0]) ** 2 + (centroid[1] - previous_position[1]) ** 2)
        if distance > abnormal_movement_threshold:
            print(f"[DEBUG] Abnormal behavior detected: movement of {distance:.2f} pixels")
            return True

    # Update the previous position
    previous_position = centroid
    return False

# Main detection loop
cropped_face_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Motion detection
    motion_detected = detect_motion(fgbg, frame)
    print(f"[DEBUG] Motion Detected: {motion_detected}")

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"[DEBUG] Faces Detected: {len(faces)}")

    # Initialize flags
    is_wearing_mask = False
    valid_conditions = 0

    if len(faces) > 0:
        valid_conditions += 1

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        save_cropped_face(face, cropped_face_count)
        cropped_face_count += 1
        is_wearing_mask = detect_mask(face)

        if is_wearing_mask:
            valid_conditions += 1

        color = (0, 255, 0) if is_wearing_mask else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = "Mask" if is_wearing_mask else "No Mask"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    confidence, weapon_detected = detect_weapon(frame)
    print(f"[DEBUG] Weapon Detected: {weapon_detected}")

    if weapon_detected:
        valid_conditions += 1

    # Check for abnormal behavior
    abnormal_behavior_detected = detect_abnormal_behavior(faces)
    if abnormal_behavior_detected:
        valid_conditions += 1

    # Debounce mechanism for alerts
    current_time = time.time()
    if valid_conditions >= 2 and (current_time - last_alert_time > alert_interval):
        print("Thief Detected! Sending alert.")
        
        # Save the frame and send alert without blocking the main loop
        threading.Thread(target=threaded_save_image, args=(frame,)).start()
        image_path = save_detected_image(frame)
        threading.Thread(target=threaded_send_alert, args=(image_path,)).start()

        # Update debounce variables
        last_alert_time = current_time
        alert_sent = True

    # Show detection statuses on the frame
    if motion_detected:
        cv2.putText(frame, "Motion Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if weapon_detected:
        cv2.putText(frame, f"Weapon Detected! ({confidence[0]:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if is_wearing_mask:
        cv2.putText(frame, "Mask Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if not motion_detected:
        cv2.putText(frame, "No Motion", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if abnormal_behavior_detected:
        cv2.putText(frame, "Abnormal Behavior Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Live Feed - Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()