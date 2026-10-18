# -*- coding: utf-8 -*-
"""Viterbi 选球放大目检：以选球点为中心裁剪 340x190 区域放大 2.2x，
画 青色候选(绿点) / YOLO球框(橙) / viterbi选球(红圈, V=真实检测 i=插值)。"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
CW, CH, ZOOM = 340, 190, 2.2

panels = []
os.makedirs(os.path.join(HERE, "vzoom"), exist_ok=True)
for si in range(len(segs)):
    dp_ = os.path.join(HERE, "dets", "%d.json" % si)
    vp = os.path.join(HERE, "dets_vit", "%d.json" % si)
    cp = os.path.join(HERE, "dets_cyan", "%d.json" % si)
    if not (os.path.exists(dp_) and os.path.exists(vp)):
        continue
    dets = json.load(open(dp_, encoding="utf-8"))
    vit = json.load(open(vp, encoding="utf-8"))
    cyans = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None
    start = segs[si][0]
    for k, fr_ in enumerate((0.15, 0.4, 0.65, 0.9)):
        idx = int(fr_ * len(vit))
        r, v = dets[idx], vit[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + r["t"]) * fps))
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        vb = v.get("ball")
        cx, cy = (int(vb["x"] * W), int(vb["y"] * H)) if vb else (W // 2, H // 2)
        x0 = max(0, min(W - CW, cx - CW // 2)); y0 = max(0, min(H - CH, cy - CH // 2))
        crop = img[y0:y0 + CH, x0:x0 + CW].copy()
        crop = cv2.resize(crop, (int(CW * ZOOM), int(CH * ZOOM)))
        sx, sy = ZOOM, ZOOM
        if cyans and idx < len(cyans):
            for c in cyans[idx]["cands"]:
                px, py = int((c["x"] * W - x0) * sx), int((c["y"] * H - y0) * sy)
                cv2.circle(crop, (px, py), 6, (0, 255, 0), 1)
        yb = r.get("ball")
        if yb:
            px, py = int((yb["x"] * W - x0) * sx), int((yb["y"] * H - y0) * sy)
            cv2.rectangle(crop, (px - 16, py - 16), (px + 16, py + 16), (0, 165, 255), 1)
        if vb:
            px, py = int((vb["x"] * W - x0) * sx), int((vb["y"] * H - y0) * sy)
            cv2.circle(crop, (px, py), 11, (0, 0, 255), 2)
            cv2.putText(crop, "V" if v.get("real") else "i", (px + 13, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(crop, "s%d t=%.1f" % (si, r["t"]), (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(crop)
        cv2.imwrite(os.path.join(HERE, "vzoom", "s%d_%d.png" % (si, k)), crop)

pw, ph = panels[0].shape[1], panels[0].shape[0]
COLS = 4
rows = []
for i in range(0, len(panels), COLS):
    row = panels[i:i + COLS]
    while len(row) < COLS:
        row.append(np.zeros((ph, pw, 3), np.uint8))
    rows.append(np.hstack(row))
grid = np.vstack(rows)
cv2.imwrite(os.path.join(HERE, "viterbi_zoom.png"), grid)
print("saved viterbi_zoom.png panels=%d size=%dx%d" % (len(panels), grid.shape[1], grid.shape[0]))
