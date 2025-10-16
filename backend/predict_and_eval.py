#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np
from pathlib import Path

def load_gt(csv_path):
    df = pd.read_csv(csv_path)
    return df.set_index('id')['N_true'].to_dict()

def count_from_result(res, mode, area_min, area_max, ratio_min):
    import numpy as np
    if mode == "seg":
        if res.masks is None: return 0
        masks = res.masks.data.cpu().numpy()
        cnt = 0
        for m in masks:
            area = m.sum()
            ys,xs = np.where(m>0)
            h,w = (ys.max()-ys.min()+1),(xs.max()-xs.min()+1)
            ratio = max(h,w)/max(1,min(h,w))
            if area_min <= area <= area_max and ratio >= ratio_min:
                cnt += 1
        return cnt
    else:
        boxes = res.boxes.data.cpu().numpy() if res.boxes is not None else []
        cnt = 0
        for b in boxes:
            x1,y1,x2,y2,conf,cls = b
            w = x2-x1; h = y2-y1
            area = w*h
            ratio = max(w,h)/max(1,min(w,h))
            if area_min <= area <= area_max and ratio >= ratio_min:
                cnt += 1
        return cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--csv_gt", required=True)
    ap.add_argument("--mode", choices=["seg","bbox"], default="seg")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--area_min", type=float, default=80)
    ap.add_argument("--area_max", type=float, default=1800)
    ap.add_argument("--ratio_min", type=float, default=1.4)
    args = ap.parse_args()

    from ultralytics import YOLO
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import pandas as pd

    gt_map = load_gt(args.csv_gt)
    model = YOLO(args.model)

    rows = []
    img_dir = Path(args.images)
    for p in sorted(img_dir.glob("*.jpg")):
        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        n_pred = count_from_result(res, args.mode, args.area_min, args.area_max, args.ratio_min)
        img_id = p.stem
        n_true = gt_map.get(img_id, None)
        rows.append({"id": img_id, "n_pred": n_pred, "n_true": n_true, "path": str(p)})

    df = pd.DataFrame(rows)
    out_csv = img_dir.parent/"predictions.csv"
    df.to_csv(out_csv, index=False)

    df2 = df.dropna()
    mae = mean_absolute_error(df2['n_true'], df2['n_pred'])
    rmse = mean_squared_error(df2['n_true'], df2['n_pred'], squared=False)
    r2 = r2_score(df2['n_true'], df2['n_pred'])
    print(f"MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
