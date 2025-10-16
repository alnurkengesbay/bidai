from ultralytics import YOLO

MODEL = r"runs/segment/train/weights/best.pt"   # если путь другой — поменяй
DATA  = r"data.yaml"                             # твой data.yaml
UNSEEN = r"data/unseen/images"                    # папка с НЕвиденными фото для глаз

m = YOLO(MODEL)

print("\n== VAL ==")
val_metrics = m.val(data=DATA, imgsz=1024, task="segment", verbose=True)
print(val_metrics)   # тут будут mAP50, mAP50-95, P/R для масок и боксов

print("\n== PREDICT (визуалки для глаз) ==")
pred = m.predict(
    source=UNSEEN,
    imgsz=1024,
    conf=0.35,
    iou=0.6,
    save=True,
    project="runs/predict",
    name="unseen16",
    verbose=True
)
print("Saved to:", pred[0].save_dir if pred else "no preds")
