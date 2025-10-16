from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # или yolov8s-seg.pt если хочешь точнее
model.train(
    data="data.yaml",
    imgsz=1024,
    epochs=40,
    batch=16,
    lr0=0.002,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.7,
    degrees=5, translate=0.05, scale=0.10, fliplr=0.5,
    mosaic=1.0, copy_paste=0.3
)
