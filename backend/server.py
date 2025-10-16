from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io, base64

# Ищем best.pt в нескольких местах
HERE = Path(__file__).parent
CANDIDATES = [
    HERE / "backend" / "weights" / "best.pt",
    HERE / "weights" / "best.pt",
    HERE / "runs" / "segment" / "train" / "weights" / "best.pt",
]
for p in CANDIDATES:
    if p.exists():
        MODEL_PATH = p
        break
else:
    raise FileNotFoundError(f"best.pt not found in: {', '.join(map(str, CANDIDATES))}")

IMGSZ_DEFAULT, CONF_DEFAULT, IOU_DEFAULT = 1024, 0.35, 0.60

app = FastAPI(title="Grain API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model():
    global model
    model = YOLO(str(MODEL_PATH))  # строка, не Path

def to_kg_per_ha(count:int, frame_side_cm:int=50, tkw_g:float=40.0)->float:
    frame_m2 = (frame_side_cm/100.0)**2
    g_per_grain = tkw_g/1000.0
    kg_m2 = (count * g_per_grain)/1000.0
    return kg_m2 * 10000.0 * frame_m2

@app.get("/health")
def health():
    return {"ok": True, "model": str(MODEL_PATH.name)}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    imgsz: int = Form(IMGSZ_DEFAULT),
    conf: float = Form(CONF_DEFAULT),
    iou: float = Form(IOU_DEFAULT),
    frame_cm: int = Form(50),
    tkw_g: float = Form(40.0),
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, task="segment", verbose=False)[0]
    vis = res.plot()[:, :, ::-1]
    count = int(res.masks.data.shape[0]) if (res.masks and res.masks.data is not None) else int(res.boxes.shape[0] if res.boxes is not None else 0)
    kg_ha = round(to_kg_per_ha(count, frame_cm, tkw_g), 2)
    buf = io.BytesIO(); Image.fromarray(vis).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"count": count, "kg_ha": kg_ha, "img_b64": f"data:image/png;base64,{b64}"}
