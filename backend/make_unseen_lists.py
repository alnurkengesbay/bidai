#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, random
from pathlib import Path

def split_names(folder, exts=('.png','.jpg','.jpeg','.webp'), ratio=0.7, seed=123):
    files = []
    for e in exts:
        files += [p.name for p in Path(folder).glob(f"*{e}")]
    files = sorted(list(set(files)))
    rnd = random.Random(seed)
    rnd.shuffle(files)
    k = int(len(files)*ratio)
    return files[:k], files[k:]

def write_list(names, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n in names: f.write(n+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprites", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    spr_tr, spr_un = split_names(args.sprites, ratio=args.train_ratio, seed=args.seed)
    bg_tr,  bg_un  = split_names(args.backgrounds, ratio=args.train_ratio, seed=args.seed)

    write_list(spr_tr, Path(args.out)/"sprites_train.txt")
    write_list(spr_un, Path(args.out)/"sprites_unseen.txt")
    write_list(bg_tr,  Path(args.out)/"backgrounds_train.txt")
    write_list(bg_un,  Path(args.out)/"backgrounds_unseen.txt")

    print("Saved lists to", args.out)

if __name__ == "__main__":
    main()
