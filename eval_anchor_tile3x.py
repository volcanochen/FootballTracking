# -*- coding: utf-8 -*-
"""实验：3x 上采样切片（更高有效分辨率）对可见锚点召回的影响。
用法: python eval_anchor_tile3x.py <weights> [conf=0.05] [fx=3.0]
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.025
NX, OV = 3, 0.20


def tile_predict(model, fr, conf, fx):
    up = cv2.resize(fr, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    tw = int(UW / (NX - (NX - 1) * OV))
    step = tw - int(tw * OV)
    out = []
    for k in range(NX):
        x0 = min(k * step, UW - tw)
        r = model.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                out.append(((float(b.xywh[0][0]) + x0) / UW, float(b.xywh[0][1]) / UH,
                            float(b.conf[0])))
    return out


def main():
    from ultralytics import YOLO
    w = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    fx = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    m = YOLO(w)
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
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
            dets = tile_predict(m, fr, conf, fx)
            best = min((math.hypot(x - b["x"], y - b["y"]) for x, y, sc in dets), default=9)
            if best < R:
                st[0] += 1
        per[si] = st
        tot += st[1]
        hit += st[0]
    cap.release()
    for k in sorted(per):
        print("seg%d: %d/%d (%.0f%%)" % (k, per[k][0], per[k][1], 100.0 * per[k][0] / max(1, per[k][1])))
    print("TILE3x fx=%.1f conf=%.2f recall: %d/%d (%.1f%%)" % (fx, conf, hit, tot, 100.0 * hit / max(1, tot)))


if __name__ == "__main__":
    main()
