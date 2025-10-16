#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_generator_scaled_v2.py
----------------------------
Случайный кроп фона + разные режимы вставки спрайтов (poisson/alpha/none),
debug-оверлеи, seed, масштаб с физикой.
"""
import argparse, random, hashlib
from pathlib import Path
import numpy as np, cv2

def set_seed(seed: int | None):
    if seed is None or seed < 0:
        return np.random.default_rng(), random
    np_rng = np.random.default_rng(seed)
    random.seed(seed)
    return np_rng, random

def load_images(dir_path, with_alpha=False):
    paths = []
    for ext in ('*.png','*.jpg','*.jpeg','*.webp'):
        paths.extend(Path(dir_path).glob(ext))
    items = []
    for p in paths:
        flag = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
        img = cv2.imread(str(p), flag)
        if img is None: 
            continue
        if with_alpha and (img.ndim == 3 or img.shape[2] == 3):
            a = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
            img = np.concatenate([img, a], axis=2)
        items.append((p.name, img))
    if not items:
        raise SystemExit(f"[ERR] Не нашёл изображений в {dir_path}")
    return items

def random_square_crop(bg, out_size, rng):
    H, W = bg.shape[:2]
    side = min(H, W)
    y0 = 0 if H == side else int(rng.integers(0, H - side + 1))
    x0 = 0 if W == side else int(rng.integers(0, W - side + 1))
    crop = bg[y0:y0+side, x0:x0+side]
    out = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5: out = cv2.flip(out, 1)
    if rng.random() < 0.5: out = cv2.flip(out, 0)
    k = int(rng.integers(0,4))
    if k: out = np.rot90(out, k).copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= float(np.clip(rng.normal(1.0, 0.08), 0.85, 1.15))
    hsv[:,:,2] *= float(np.clip(rng.normal(1.0, 0.10), 0.85, 1.20))
    out = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def rect_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = aw*ah + bw*bh - inter + 1e-9
    return inter/ua

class LayoutEngine:
    def __init__(self, W, H, iou_max=0.15):
        self.W, self.H = W, H
        self.boxes = []
        self.iou_max = iou_max
    def can_place(self, rect):
        return all(rect_iou(r, rect) <= self.iou_max for r in self.boxes)
    def add(self, rect):
        self.boxes.append(rect)

def per_sprite_color_jitter(rng, sprite_rgba):
    rgb = sprite_rgba[:, :, :3].astype(np.float32)
    a = sprite_rgba[:, :, 3:4]
    alpha = float(np.clip(rng.normal(1.0, 0.10), 0.85, 1.15))
    beta  = float(np.clip(rng.normal(0.0, 12.0), -18, 18))
    rgb = np.clip(rgb*alpha + beta, 0, 255)
    gamma = float(np.clip(rng.normal(1.0, 0.06), 0.9, 1.1))
    rgb = np.clip(255.0 * ((rgb/255.0) ** (1.0/gamma)), 0, 255)
    return np.concatenate([rgb.astype(np.uint8), a], axis=2)

def resize_longside_to(sprite_rgba, target_long_px):
    h, w = sprite_rgba.shape[:2]
    longside = max(h, w)
    if longside <= 0: return sprite_rgba
    scale = float(target_long_px) / float(longside)
    nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
    return cv2.resize(sprite_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

def mix_alpha(bg_roi, spr_rgb, mask_bin):
    alpha = (mask_bin.astype(np.float32) / 255.0)[...,None]
    return (spr_rgb*alpha + bg_roi*(1.0-alpha)).astype(np.uint8)

def paste_sprite(bg, spr_rgba, x, y, blend_mode='poisson', shadow_strength=0.18, shadow_dir=(1.0,0.35)):
    H, W = bg.shape[:2]
    h, w = spr_rgba.shape[:2]
    if x < 0 or y < 0 or x+w > W or y+h > H:
        return None, None, None

    spr_rgb = spr_rgba[:,:,:3].copy()
    alpha   = spr_rgba[:,:,3].copy()

    alpha = cv2.GaussianBlur(alpha, (3,3), 0)
    alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    mask  = (alpha > 8).astype(np.uint8) * 255

    roi = bg[y:y+h, x:x+w]

    if shadow_strength > 0:
        sx = int(shadow_dir[0]*3); sy = int(shadow_dir[1]*3)
        sh = np.zeros_like(mask)
        M = np.float32([[1,0,sx],[0,1,sy]])
        cv2.warpAffine(mask, M, (w,h), dst=sh, flags=cv2.INTER_NEAREST, borderValue=0)
        sh = cv2.GaussianBlur(sh, (7,7), 0)
        dark = int(shadow_strength*90)
        roi[:] = cv2.subtract(roi, cv2.cvtColor((sh*(dark/255)).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    if blend_mode == 'poisson':
        center = (x + w//2, y + h//2)
        blended = cv2.seamlessClone(spr_rgb, bg, mask, center, cv2.MIXED_CLONE)
        bg[:] = blended
    elif blend_mode == 'alpha':
        roi[:] = mix_alpha(roi, spr_rgb, mask)
    else:
        spr_rgb2 = spr_rgb.copy()
        spr_rgb2[mask==0] = roi[mask==0]
        roi[:] = spr_rgb2

    return bg, (x, y, w, h), (mask > 0).astype(np.uint8)

def global_augs(rng, img):
    img = cv2.GaussianBlur(img, (3,3), 0)
    q = int(np.clip(rng.normal(85, 8), 62, 95))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok: img = cv2.imdecode(enc, 1)
    return img

def yolo_bbox_line(cls, x, y, w, h, W, H):
    cx = (x + w/2) / W; cy = (y + h/2) / H
    bw = w / W; bh = h / H
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def yolo_seg_line(cls, poly, W, H):
    coords = []
    for x, y in poly:
        coords.append(f"{x / W:.6f}"); coords.append(f"{y / H:.6f}")
    return f"{cls} " + " ".join(coords)

def poly_from_mask(mask, simplify_eps=1.5):
    cnts, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if c.shape[0] < 3: continue
        c = cv2.approxPolyDP(c, simplify_eps, True)
        polys.append(c.reshape(-1,2))
    return polys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprites", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--mode", choices=["bbox","seg"], default="seg")
    ap.add_argument("--kmin", type=int, default=25)
    ap.add_argument("--kmax", type=int, default=70)
    ap.add_argument("--overlap_max", type=float, default=0.15)
    ap.add_argument("--shadow", type=float, default=0.18)
    ap.add_argument("--frame_cm", type=float, default=50.0)
    ap.add_argument("--grain_mm_min", type=float, default=6.0)
    ap.add_argument("--grain_mm_max", type=float, default=8.0)
    ap.add_argument("--size_jitter", type=float, default=0.12)
    ap.add_argument("--min_long_px", type=int, default=8)
    ap.add_argument("--scale_mul", type=float, default=1.0)
    ap.add_argument("--neg_prob", type=float, default=0.1)
    ap.add_argument("--blend", choices=["poisson","alpha","none"], default="poisson")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    rng, pyrand = set_seed(args.seed)
    sprites = load_images(args.sprites, with_alpha=True)
    backs   = load_images(args.backgrounds, with_alpha=False)

    out_im = Path(args.out) / "images"
    out_lb = Path(args.out) / "labels"
    out_dbg = Path(args.out) / "debug"
    out_im.mkdir(parents=True, exist_ok=True)
    out_lb.mkdir(parents=True, exist_ok=True)
    if args.debug: out_dbg.mkdir(parents=True, exist_ok=True)

    px_per_cm = args.size / float(args.frame_cm)

    for i in range(args.n):
        _, bg0 = pyrand.choice(backs)
        bg = random_square_crop(bg0, args.size, rng)
        H, W = bg.shape[:2]

        if rng.random() < max(0.0, min(1.0, args.neg_prob)):
            bg2 = global_augs(rng, bg.copy())
            cv2.imwrite(str(out_im / f"im_{i:05d}.jpg"), bg2)
            (out_lb / f"im_{i:05d}.txt").write_text("")
            if args.debug:
                cv2.imwrite(str(out_dbg / f"im_{i:05d}_NEG.jpg"), bg2)
            continue

        eng = LayoutEngine(W, H, iou_max=args.overlap_max)
        K = pyrand.randint(args.kmin, args.kmax+1)
        labels_lines, seg_lines = [], []
        attempts, attempts_limit, placed = 0, K*60, 0
        dbg_img = bg.copy()

        while placed < K and attempts < attempts_limit:
            attempts += 1
            _, spr_rgba = pyrand.choice(sprites)
            spr_rgba = per_sprite_color_jitter(rng, spr_rgba)

            grain_mm = float(rng.uniform(args.grain_mm_min, args.grain_mm_max))
            target_long_px = (grain_mm/10.0) * px_per_cm * args.scale_mul
            target_long_px *= (1.0 + float(rng.uniform(-args.size_jitter, args.size_jitter)))
            target_long_px = max(args.min_long_px, target_long_px)
            spr = resize_longside_to(spr_rgba, target_long_px)

            angle = float(rng.uniform(0, 180))
            M = cv2.getRotationMatrix2D((spr.shape[1]/2, spr.shape[0]/2), angle, 1.0)
            spr = cv2.warpAffine(spr, M, (spr.shape[1], spr.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))

            h, w = spr.shape[:2]
            if h < args.min_long_px or w < args.min_long_px:
                continue

            x = int(rng.integers(0, W - w))
            y = int(rng.integers(0, H - h))
            rect = (x, y, w, h)
            if not eng.can_place(rect): 
                continue

            pasted = paste_sprite(bg, spr, x, y, blend_mode=args.blend, shadow_strength=args.shadow, shadow_dir=(1.0,0.35))
            if pasted[0] is None:
                continue

            bg, rect, mask = pasted
            eng.add(rect); placed += 1

            if args.debug:
                cv2.rectangle(dbg_img, (rect[0],rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 1)

            if args.mode == "bbox":
                labels_lines.append(yolo_bbox_line(0, rect[0], rect[1], rect[2], rect[3], W, H))
            else:
                for P in poly_from_mask(mask):
                    if P.shape[0] < 3: continue
                    P[:,0] += rect[0]; P[:,1] += rect[1]
                    seg_lines.append(yolo_seg_line(0, P, W, H))

        bg2 = global_augs(rng, bg)
        im_path = out_im / f"im_{i:05d}.jpg"
        lb_path = out_lb / f"im_{i:05d}.txt"
        cv2.imwrite(str(im_path), bg2)
        if args.mode == "bbox":
            (lb_path).write_text("\n".join(labels_lines))
        else:
            (lb_path).write_text("\n".join(seg_lines))
        if args.debug:
            dbg = cv2.addWeighted(dbg_img, 0.6, bg2, 0.4, 0)
            cv2.putText(dbg, f"placed:{placed}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.imwrite(str(out_dbg / f"im_{i:05d}_DBG.jpg"), dbg)

    print("Done. Images -> images/, labels -> labels/, debug -> debug/ (если включен).")

if __name__ == "__main__":
    main()
