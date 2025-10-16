import type { PredictOut } from "./types";

const BASE = import.meta.env.VITE_API_URL as string;

export type PredictParams = {
  model_path: string;
  imgsz: number;
  conf: number;
  iou: number;
  area_min: number;
  area_max: number;
  ratio_min: number;
  ratio_max: number;
  frame_cm: number;
  tkw: number;
};

export async function health(model_path: string) {
  const u = new URL("/health", BASE);
  u.searchParams.set("model_path", model_path);
  const r = await fetch(u);
  if (!r.ok) throw new Error("health failed");
  return r.json();
}

export async function predictOne(file: File, p: PredictParams): Promise<PredictOut> {
  const u = new URL("/predict", BASE);
  for (const [k, v] of Object.entries(p)) u.searchParams.set(k, String(v));
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(u, { method: "POST", body: fd });
  if (!r.ok) throw new Error("predict failed");
  return r.json();
}
