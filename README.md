# bid.ai — Grain counting demo

This repository contains a FastAPI backend that runs a YOLOv8 segmentation model and a small Vite + React frontend in `grain-ui/` for uploading images and getting grain counts and overlay visualizations.

Quick local run (Windows PowerShell)

1. Create and activate a virtualenv

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install Python deps

```powershell
pip install -r requirements.txt
```

3. Start backend (from project root)

```powershell
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

4. Start frontend

```powershell
cd grain-ui
npm install
npm run dev
```

Open the Vite URL shown in the terminal and use the UI.

Deployment notes

- Frontend is a static Vite app, deployable to Vercel or Netlify. In Vercel point it at `grain-ui` and set build command `npm run build` and output directory `dist` (or let Vercel auto-detect).
- Backend (FastAPI) can be deployed to Render, Fly, or any server supporting Python. Add `requirements.txt` and ensure the model file `runs/segment/train/weights/best.pt` or `yolov8n-seg.pt` is available on the server.

Optional: Dockerize backend (not included here).

If you want, I can add GitHub Actions to auto-deploy frontend + backend.

Notes about model and GPU

- The YOLOv8 model used for segmentation may be large. If you use a CPU-only host, inference will be slower. For reasonable latency consider using a GPU-backed instance (e.g., Render's GPU plans or any cloud VM with CUDA and a GPU).

Docker / Render quick deploy

- Build locally: docker build -t bidai-backend .
- Run: docker run -p 8000:8000 bidai-backend

