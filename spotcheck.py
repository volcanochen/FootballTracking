# -*- coding: utf-8 -*-
"""抽查 v2：每个抽帧 = 全帧(缩放0.6) + 圈位 4x 放大插图，纵向排列。"""
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 29.97
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))

CHECKS = {
    0: [10, 40, 70, 100],
    5: [5, 25, 50, 70],
    7: [10, 40, 80, 120],
    2: [15, 55, 95, 135],
}

for si, idxs in CHECKS.items():
    tr = json.load(open(os.path.join(HERE, "dets_track", "%d.json" % si), encoding="utf-8"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
    want = {3 * i: i for i in idxs}
    tiles = []
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f in want:
            i = want[f]
            b = tr[i].get("ball") if i < len(tr) else None
            tag = "s%d i%d none" % (si, i)
            if b:
                px, py = int(b["x"] * 854), int(b["y"] * 356)
                src = tr[i].get("src", "?")
                col = {"A": (0, 220, 0), "F": (0, 0, 255), "B": (0, 0, 255),
                       "S": (0, 0, 255), "w": (0, 230, 230), "I": (0, 150, 255)}.get(src, (255, 255, 255))
                cv2.circle(fr, (px, py), 12, col, 2)
                tag = "s%d i%d %s" % (si, i, src)
            cv2.putText(fr, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            half = cv2.resize(fr, (512, 213))
            if b:
                R = 40
                cx, cy = int(b["x"] * 854), int(b["y"] * 356)
                t = fr[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
                if t.size:
                    t = cv2.resize(t, (R * 8, R * 8), interpolation=cv2.INTER_NEAREST)
                    tt = cv2.resize(t, (213, 213))
                    half = cv2.hconcat([half, tt])
            tiles.append(half)
        f += 1
        if f > max(want) + 1:
            break
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        canvas = cv2.vconcat(tiles)
        out = os.path.join(HERE, "spotcheck_%d.jpg" % si)
        cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("saved", out, flush=True)
cap.release()
print("done")
