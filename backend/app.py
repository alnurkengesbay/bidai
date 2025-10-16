import os, io, time, base64
from typing import List, Tuple
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Ultralytics не установлен: {e}")

# ---------- CORS ----------
ALLOWED = [
    "http://localhost:5173",     # streamlit локально
    "http://localhost:7860",
    "https://<твой_streamlit_домен>",  # добавишь когда задеплоишь
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- МОДЕЛИ ----------
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "backend/weights/yolov8n-seg.pt")

@lru_cache(maxsize=4)
def load_model_cached(path: str):
    return YOLO(path)

def to_png_b64(np_bgr) -> str:
    # np_bgr -> PNG base64
    rgb = np_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def post_filter_masks(r, area_min:int, area_max:int, ratio_min:float, ratio_max:float) -> Tuple[int,int]:
    if r is None or r.masks is None or r.masks.data is None:
        return 0, 0
    m = r.masks.data  # [N,H,W] torch
    raw = int(m.shape[0])
    mask_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
    keep = []
    for i in range(raw):
        mask = mask_np[i]
        area = int(mask.sum())
        if area < area_min or area > area_max: 
            continue
        ys, xs = np.where(mask > 0)
        if xs.size == 0: 
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        w = max(1, x2 - x1 + 1)
        h = max(1, y2 - y1 + 1)
        aspect = max(w/h, h/w)
        if ratio_min <= aspect <= ratio_max:
            keep.append(i)
    return raw, len(keep)

def to_kg_per_ha(count:int, frame_side_cm:int, tkw_g:float) -> float:
    frame_m2 = (frame_side_cm/100.0) ** 2
    g_per_grain = tkw_g / 1000.0
    kg_per_m2 = (count * g_per_grain) / 1000.0
    return kg_per_m2 * 10000.0 * frame_m2

class PredictOut(BaseModel):
    raw: int
    kept: int
    kg_ha: float
    ms: int
    boxes: List[List[float]]
    image_png_b64: str

@app.get("/health")
def health(model_path: str = Query(DEFAULT_MODEL_PATH)):
    return {"ok": True, "model": os.path.basename(model_path)}

@app.post("/predict", response_model=PredictOut)
async def predict(
    file: UploadFile = File(...),
    model_path: str = Query(DEFAULT_MODEL_PATH),
    imgsz: int = Query(1024, ge=320, le=2048),
    conf: float = Query(0.35, ge=0.01, le=0.99),
    iou: float = Query(0.6, ge=0.1, le=0.95),
    area_min: int = Query(25, ge=0),
    area_max: int = Query(5000, ge=1),
    ratio_min: float = Query(1.4, ge=1.0),
    ratio_max: float = Query(4.0, ge=1.0),
    frame_cm: int = Query(50, ge=1),
    tkw: float = Query(40.0, ge=1.0)
):
    img_bytes = await file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    model = load_model_cached(model_path)
    t0 = time.time()
    res = model.predict(pil, imgsz=imgsz, conf=conf, iou=iou, task="segment", verbose=False)
    r = res[0]

    # boxes
    boxes = []
    if r.boxes is not None:
        for b in r.boxes.xyxy.cpu().tolist():
            x1,y1,x2,y2 = map(float, b)
            boxes.append([x1,y1,x2,y2])

    raw, kept = post_filter_masks(r, area_min, area_max, ratio_min, ratio_max)

    # визуалка (берём готовый plot() и подпишем параметры)
    vis_np_bgr = r.plot()
    # подпись
    try:
        from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA
        import cv2
        rectangle(vis_np_bgr, (18,18), (18+460, 18+46), (10,24,60), -1)
        putText(vis_np_bgr, f"raw:{raw} kept:{kept} conf={conf:.2f} iou={iou:.2f} imgsz={imgsz}",
                (28, 50), FONT_HERSHEY_SIMPLEX, 0.9, (240,240,255), 2, LINE_AA)
    except Exception:
        # fallback подпись через PIL
        pil_vis = Image.fromarray(vis_np_bgr[:, :, ::-1])
        draw = ImageDraw.Draw(pil_vis)
        draw.text((20, 20), f"raw:{raw} kept:{kept} conf={conf:.2f} iou={iou:.2f} imgsz={imgsz}", fill=(255,255,255))
        vis_np_bgr = np.asarray(pil_vis)[:, :, ::-1]

    kg = to_kg_per_ha(kept, frame_cm, tkw)
    ms = int((time.time() - t0) * 1000)

    return PredictOut(
        raw=int(raw),
        kept=int(kept),
        kg_ha=float(kg),
        ms=ms,
        boxes=boxes,
        image_png_b64=to_png_b64(vis_np_bgr),
    )
