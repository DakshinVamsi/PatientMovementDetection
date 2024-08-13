if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO(r"C:\Users\daksh\Vamsi Python\Deep Learning\runs\detect\train6\weights\best.pt")  # Update with the correct path to your trained model

    model.train(
        data="config.yaml",
        epochs=5,
        device=0,
        lr0=0.01,
        lrf=0.0001,
        augment=True,
        batch=32,
        imgsz=640,
        patience=5,
        weight_decay=0.0005,
    )
