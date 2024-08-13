import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\daksh\Vamsi Python\Deep Learning\runs\detect\train11\weights\best.pt") 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model.predict(source=frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy() 
        class_ids = result.boxes.cls.cpu().numpy()  

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{model.names[int(class_id)]} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Patient Movement Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
