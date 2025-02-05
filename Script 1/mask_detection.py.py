import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('mask_detector.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the required format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (62, 62))  # Adjusted size
    img = img / 255.0  # Normalize if needed
    img = np.expand_dims(img, axis=0)

    # Predict mask or no mask
    prediction = model.predict(img)
    print("Prediction Raw Output:", prediction)  # Debugging output

    # Adjust threshold if needed
    label = "Mask" if prediction[0][0] > 0.7 else "No Mask"
    
    # Display the label on the image
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Mask Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cls