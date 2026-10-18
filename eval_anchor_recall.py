# -*- coding: utf-8 -*-
"""在原管线人工确认的轨迹锚点（dets_track real=True，即球确定可见的帧）上
评测切片模型检出率——这才是"识别率"的正确分母。
用法: python eval_anchor_recall.py <weights> [conf=0.10]
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.02
NX, OV = 3, 0.20


def tile_predict(model, fr, conf):
    H, W = fr.shape[:2]
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    tw = int(UW / (NX - (NX - 1) * OV))
    step = tw - int(tw * OV)
    out = []
    for k in range(NX):
        x0 = min(k * step, UW - tw)
        r = model.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                continue
            bx, by = float(b.xywh[0][0]), float(b.xywh[0][1])
            out.append(((bx + x0) / UW, by / UH, float(b.conf[0])))
    return out


def main():
    from ultralytics import YOLO
    weights = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    model = YOLO(weights)
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    tot = hit = 0
    per = {}
    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        if not os.path.exists(dp):
            continue
        d = json.load(open(dp, encoding="utf-8"))
        st = [0, 0]
        for a, rec in enumerate(d):
            b = rec.get("ball")
            if not b or not rec.get("real"):
                continue
            fi = int(segs[si][0] * fps) + a * 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            st[1] += 1
            best = None
            for (x, y, sc) in tile_predict(model, fr, conf):
                dist = math.hypot(x - b["x"], y - b["y"])
                if best is None or dist < best[0]:
                    best = (dist, sc)
            if best and best[0] < R:
                st[0] += 1
        per[si] = st
        tot += st[1]
        hit += st[0]
    cap.release()
    for k in sorted(per):
        print("seg%d: %d/%d (%.0f%%)" % (k, per[k][0], per[k][1], 100.0 * per[k][0] / max(1, per[k][1])))
    print("TOTAL visible-anchor recall: %d/%d (%.1f%%)" % (hit, tot, 100.0 * hit / max(1, tot)))


if __name__ == "__main__":
    main()
