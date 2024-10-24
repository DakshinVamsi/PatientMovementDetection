Real-Time Patient Movement Detection with YOLOv5
This project aims to develop a real-time monitoring system to detect critical patient movements, such as attempts to get out of bed, using the YOLOv5 object detection model. It leverages custom video data for healthcare monitoring and alert generation.

Features
Real-Time Detection: Monitors and detects patient movements in real-time through live video feeds.
Custom Dataset: Annotated with CVAT and organized in YOLO format for model training.
Movement Labels: Detection of specific movements categorized as idle patient, non_trigger_movement, and trigger_movement.
Streamlit Interface: Provides a user-friendly dashboard for uploading videos and live monitoring.
Alert System: Displays a notification when the model detects critical movements, such as attempts to get out of bed.

Technologies Used
YOLOv5: State-of-the-art object detection model, implemented using PyTorch.
Python: Core language used for model training, dataset management, and real-time inference.
CVAT (Computer Vision Annotation Tool): Used to annotate the dataset with bounding boxes and labels.
Streamlit: Provides an interactive interface for real-time video monitoring and alerts.
OpenCV: For real-time video processing and drawing bounding boxes around detected movements.
PyTorch: Deep learning framework for training and fine-tuning the YOLOv5 model.
