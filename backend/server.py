# pip install fastapi uvicorn[standard] ultralytics pillow numpy opencv-python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import numpy as np, io, base64

MODEL_PATH = r"runs/segment/train/weights/best.pt"
IMGSZ_DEFAULT, CONF_DEFAULT, IOU_DEFAULT = 1024, 0.35, 0.60

app = FastAPI(title="Grain API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

model = YOLO(MODEL_PATH)

def to_kg_per_ha(count:int, frame_side_cm:int=50, tkw_g:float=40.0)->float:
    frame_m2 = (frame_side_cm/100.0)**2
    g_per_grain = tkw_g/1000.0
    kg_m2 = (count * g_per_grain)/1000.0
    return kg_m2 * 10000.0 * frame_m2

@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    imgsz: int = Form(IMGSZ_DEFAULT),
    conf: float = Form(CONF_DEFAULT),
    iou: float = Form(IOU_DEFAULT),
    frame_cm: int = Form(50),
    tkw_g: float = Form(40.0),
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    res = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, task="segment", verbose=False)[0]

    # визуализация
    vis = res.plot()[:, :, ::-1]  # BGR->RGB
    # счёт по маскам
    count = int(res.masks.data.shape[0]) if (res.masks and res.masks.data is not None) else int(res.boxes.shape[0] if res.boxes is not None else 0)
    kg_ha = round(to_kg_per_ha(count, frame_cm, tkw_g), 2)

    # кодируем картинку
    out = io.BytesIO(); Image.fromarray(vis).save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return {"count": count, "kg_ha": kg_ha, "img_b64": f"data:image/png;base64,{b64}"}
