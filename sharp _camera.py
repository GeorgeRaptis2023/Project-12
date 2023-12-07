import cv2
import time
import numpy as np

# Open a connection to the camera (usually 0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Add a half-second delay
time.sleep(0.5)

# Capture a single frame
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not read frame.")
    exit()

# Define a contrast factor (1.0 is the original contrast)
contrast_factor = 2.0  # Increase contrast by a factor of 10

brightness_factor = 60.0  # Increase brightness by a factor of 100

# Adjust the contrast of the frame
increased_contrast_frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)

# Adjust the brightness of the frame
brightened_frame = cv2.convertScaleAbs(increased_contrast_frame, alpha=1.0, beta=brightness_factor)

# Define a sharpening kernel
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])

# Apply the sharpening filter
sharpened_frame = cv2.filter2D(brightened_frame, -1, sharpening_kernel)

# Save the frame with increased contrast, brightness, and sharpening to an image file
cv2.imwrite("sharpened_image.jpg", sharpened_frame)

# Release the camera
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
