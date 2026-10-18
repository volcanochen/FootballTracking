# -*- coding: utf-8 -*-
"""切片放大推理评测：把 2x 上采样帧按左右 3 切片（各 1280 宽），有效分辨率 ~4.5x。
对比全帧推理的种子召回。用法: python eval_seeds_tile.py <weights> [conf=0.10]
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.015
NX = 3          # 横向切片数
OV = 0.20       # 切片间重叠比例


def tile_predict(model, fr, conf):
    """返回归一化(原图坐标系)的 [(x,y,w,h,conf)]，NMS 合并各切片。"""
    H, W = fr.shape[:2]
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    tw = int(UW / (NX - (NX - 1) * OV))
    step = tw - int(tw * OV)
    out = []
    for k in range(NX):
        x0 = min(k * step, UW - tw)
        tile = up[:, x0:x0 + tw]
        r = model.predict(tile, imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                continue
            bx, by, bw, bh = [float(v) for v in b.xywh[0]]   # tile 内像素
            gx = (bx + x0) / UW               # -> 2x 图归一化
            gy = by / UH
            gw, gh = bw / UW, bh / UH
            out.append((gx, gy, gw, gh, float(b.conf[0])))
    # 简易 NMS
    out.sort(key=lambda d: -d[4])
    keep = []
    for c in out:
        ok = True
        for k in keep:
            ix = max(0, min(c[0] + c[2] / 2, k[0] + k[2] / 2) - max(c[0] - c[2] / 2, k[0] - k[2] / 2))
            iy = max(0, min(c[1] + c[3] / 2, k[1] + k[3] / 2) - max(c[1] - c[3] / 2, k[1] - k[3] / 2))
            inter = ix * iy
            uni = c[2] * c[3] + k[2] * k[3] - inter
            if uni > 0 and inter / uni > 0.45:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def main():
    from ultralytics import YOLO
    weights = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    seeds = json.load(open(os.path.join(HERE, "seeds_v2.json"), encoding="utf-8"))
    rv = json.load(open(os.path.join(HERE, "review_result.json"), encoding="utf-8"))
    bad = set(rv.get("badBoxes", []))
    good = [s for s in seeds if "s%d_i%04d#S0" % (s["seg"], s["i"]) not in bad]
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    model = YOLO(weights)
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    hit = hit_hi = 0
    miss = []
    for s in good:
        fi = int(segs[s["seg"]][0] * fps) + s["i"] * 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        dets = tile_predict(model, fr, conf)
        best = None
        for d in dets:
            dist = math.hypot(d[0] - s["x"], d[1] - s["y"])
            if best is None or dist < best[0]:
                best = (dist, d[4])
        if best and best[0] < R:
            hit += 1
            if best[1] > 0.5:
                hit_hi += 1
        else:
            miss.append("s%d_i%d(%s)" % (s["seg"], s["i"],
                        round(best[0], 3) if best else "none"))
    cap.release()
    print("TILE conf=%.2f  recall=%d/%d (%.1f%%)  hiConf=%d" % (
        conf, hit, len(good), 100.0 * hit / max(1, len(good)), hit_hi))
    print("miss:", "; ".join(miss))


if __name__ == "__main__":
    main()
