import cv2
import numpy as np
from datetime import datetime

class LettuceTracker:
    def __init__(self):
        self.stable_frame_count = 0
        self.stable_height = None
        self.stable_bbox = None
        self.unstable_frame_count = 0

    def calculate_height(self, contour):
        # Assuming the lettuce plant is relatively straight, use the bounding box height as an estimate
        x, y, w, h = cv2.boundingRect(contour)
        return h

    def is_lettuce(self, contour):
        # Adjust these parameters based on the characteristics of your lettuce plants
        aspect_ratio_min = 0.5
        aspect_ratio_max = 2.0
        area_min = 500

        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Check aspect ratio and area to filter out non-lettuce contours
        return aspect_ratio_min < aspect_ratio < aspect_ratio_max and cv2.contourArea(contour) > area_min

    def update(self, frame, contours):
        stable_detected = False

        for contour in contours:
            if self.is_lettuce(contour):
                height = self.calculate_height(contour)

                # Check if the current frame is consistent with the stable frame
                if self.stable_frame_count > 0 and abs(height - self.stable_height) < 10:
                    self.stable_frame_count += 1
                    if self.stable_frame_count >= 5:  # Adjust stability threshold as needed
                        stable_detected = True
                        self.stable_bbox = cv2.boundingRect(contour)
                        break
                else:
                    self.stable_height = height
                    self.stable_frame_count = 1

        if not stable_detected:
            self.stable_frame_count = 0

        return stable_detected

    def draw_stable_bbox(self, frame):
        if self.stable_bbox:
            x, y, w, h = self.stable_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Stable Height: {self.stable_height}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def process_frame(frame, tracker):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for stability and update tracker
    stable_detected = tracker.update(frame, contours)

    # Draw bounding box only if lettuce is stable
    if stable_detected:
        tracker.draw_stable_bbox(frame)

    # Add timestamp to the frame
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Open the video capture (you may need to adjust the argument to your camera index or video file)
cap = cv2.VideoCapture(0)

lettuce_tracker = LettuceTracker()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Process the frame
    processed_frame = process_frame(frame, lettuce_tracker)

    # Display the processed frame
    cv2.imshow('Lettuce Height Detection', processed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()