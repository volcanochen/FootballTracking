# -*- coding: utf-8 -*-
"""白色球候选 + YOLO ballAll 触表（第二轮标注，针对真球=白色小球）。
排除固定青色光源 (0.630, 0.565)。
白球特征：高亮度(val>175)、低饱和(sat<90)、圆形小块(area 2~40)。
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
LIGHT = (0.630, 0.565)     # 固定青色光源（26 个误标注证实）
SKIP = 6
TILE = 20
COLS = 8

os.makedirs(os.path.join(HERE, "sheets2"), exist_ok=True)


def white_candidates(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 175), (180, 90, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H, W = bgr.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 2 <= area <= 40 and w <= 10 and h <= 10 and 0.4 <= w / max(1, h) <= 2.5:
            # 排除白线：白线是长条，连通域窄长已被 w/h 限制；再排除贴着长结构的情况
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            if abs(cx - LIGHT[0]) < 0.03 and abs(cy - LIGHT[1]) < 0.03:
                continue
            out.append({"x": cx, "y": cy, "area": int(area)})
    return out


cap = cv2.VideoCapture(VIDEO)
tiles = []
meta = {}
tid = 0
for si in range(8):
    dp_ = os.path.join(HERE, "dets", "%d.json" % si)
    if not os.path.exists(dp_):
        continue
    dets = json.load(open(dp_, encoding="utf-8"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
    fi_video = 0
    idx = 0
    while idx < len(dets):
        ok, fr = cap.read()
        if not ok:
            break
        fi_video += 1
        if fi_video - 1 != idx * 3:
            continue
        if idx % SKIP == 0:
            H, W = fr.shape[:2]
            cands = []
            for b in (dets[idx].get("ballAll") or []):
                cands.append((b["x"], b["y"], "yolo%.2f" % b["score"]))
            for c in white_candidates(fr):
                cands.append((c["x"], c["y"], "white%d" % c["area"]))
            dedup = []
            for c in cands:
                if not any(abs(c[0] - d[0]) < 0.02 and abs(c[1] - d[1]) < 0.02 for d in dedup):
                    dedup.append(c)
            for (x, y, src) in dedup:
                px, py = int(x * W), int(y * H)
                if not (TILE < px < W - TILE and TILE < py < H - TILE):
                    continue
                tile = cv2.resize(fr[py - TILE:py + TILE, px - TILE:px + TILE], (160, 160),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile, "#%d" % tid, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tiles.append(tile)
                meta[str(tid)] = {"seg": si, "i": idx, "t": dets[idx]["t"], "x": round(x, 4),
                                  "y": round(y, 4), "src": src}
                tid += 1
        idx += 1
cap.release()
for i in range(0, len(tiles), COLS):
    row = tiles[i:i + COLS]
    while len(row) < COLS:
        row.append(np.zeros((160, 160, 3), np.uint8))
    cv2.imwrite(os.path.join(HERE, "sheets2", "t_%03d.png" % (i // COLS)), np.hstack(row))
with open(os.path.join(HERE, "sheets2", "meta.json"), "w", encoding="utf-8") as fp:
    json.dump(meta, fp, ensure_ascii=False, indent=1)
from collections import Counter
print("tiles=%d sheets=%d" % (len(tiles), (len(tiles) + COLS - 1) // COLS))
print("src:", dict(Counter(v.split(".")[0].rstrip("0123456789") if "." in v else v for _, v in [ (k, m2["src"]) for k, m2 in meta.items() ])))
