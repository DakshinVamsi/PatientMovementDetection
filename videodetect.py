import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO(r"C:\Users\daksh\Vamsi Python\Deep Learning\runs\detect\train11\weights\best.pt")  # Update with the path to your trained model

# Path to the video file
video_path = r"C:\Users\daksh\OneDrive\Desktop\DL project - patient\Test Videos\CloseUpVideos_Right\6.mkv"  # Update with the path to your video file

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the labels with swapped values
label_map = {
    1: 'trigger_movement',       # assuming class ID 1 is 'non_trigger_movement' but we swap it to 'trigger_movement'
    2: 'non_trigger_movement'    # assuming class ID 2 is 'trigger_movement' but we swap it to 'non_trigger_movement'
}

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Use the model to make predictions
    results = model.predict(source=frame)

    # Flag to check if non_trigger_movement is detected
    trigger_movement_detected = False

    # Loop through results and draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Extract confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Extract class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = label_map.get(int(class_id), 'unknown')

            # Check for trigger_movement
            if label == 'trigger_movement':
                trigger_movement_detected = True

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display warning text if trigger_movement is detected
    if trigger_movement_detected:
        cv2.putText(frame, 'Patient is trying to get out of the bed!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Display the frame with the detections
    cv2.imshow('Patient Movement Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
