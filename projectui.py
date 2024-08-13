import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

model = YOLO(r"C:\Users\daksh\Vamsi Python\Deep Learning\runs\detect\train11\weights\best.pt")

label_map = {
    1: 'trigger_movement',       # assuming class ID 1 is 'trigger_movement'
    2: 'non_trigger_movement'    # assuming class ID 2 is 'non_trigger_movement'
}

st.title('Patient Movement Detection')

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Process the video
    cap = cv2.VideoCapture(tmp_file_path)
    
    stframe = st.empty()  # Placeholder for video frame
    stmessage = st.empty()  # Placeholder for message

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8 inference
        results = model.predict(source=frame)
        
        trigger_movement_detected = False
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

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
            stmessage.text('Patient is trying to get out of bed!')
        else:
            stmessage.text('')  # Clear the message if no trigger movement detected

        # Display the frame with bounding boxes
        stframe.image(frame, channels="BGR")
    
    cap.release()
    
    # Clean up the temporary file
    os.remove(tmp_file_path)