from ultralytics import YOLO
import numpy as np
import gradio as gr

# === настройки ===
MODEL_PATH = r"runs/segment/train/weights/best.pt"
IMGSZ = 1024
CONF  = 0.35
IOU   = 0.60

# грузим модель один раз
model = YOLO(MODEL_PATH)

def infer(image):
    # image: PIL.Image или ndarray — Gradio сам приведёт
    res = model.predict(image, imgsz=IMGSZ, conf=CONF, iou=IOU, task="segment", verbose=False)
    if not res or res[0] is None:
        return image, 0, "нет предсказаний"

    r = res[0]
    # картинка с наложенными масками/боксами
    vis = r.plot()                   # numpy (H, W, 3) BGR
    vis = vis[:, :, ::-1].copy()     # BGR->RGB

    # счёт объектов (масок)
    if r.masks is not None and r.masks.data is not None:
        count = int(r.masks.data.shape[0])
    else:
        count = int(r.boxes.shape[0]) if r.boxes is not None else 0

    # сюда можно добавить свои post-фильтры по площади/форме, если нужно
    return vis, count, f"imgsz={IMGSZ}, conf={CONF}, iou={IOU}"

title = "Grain Loss MVP (YOLOv8-seg)"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"## {title}\nЗагрузи фото рамки 50×50 см — получишь маски и счёт зёрен.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Фото (jpg/png)")
        out_img = gr.Image(type="numpy", label="Предсказание", elem_id="pred")
    with gr.Row():
        out_count = gr.Number(label="Зёрен на фото", precision=0)
        out_info  = gr.Textbox(label="Параметры инференса", interactive=False)
    btn = gr.Button("Рассчитать")
    btn.click(infer, inputs=inp, outputs=[out_img, out_count, out_info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # локальная сеть тоже увидит
