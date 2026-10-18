# -*- coding: utf-8 -*-
"""轨迹质量验证：
1) tracks 焦点 vs YOLO 球位置的平均距离（有球帧）
2) 可视化：原帧 + YOLO 球框(橙) + tracks 焦点圈(青) + 放大示意框
"""
import json
import os

import cv2
import numpy as np


def np_vstack_rows(rows):
    return np.vstack([np.hstack(r) for r in rows])

HERE = os.path.dirname(os.path.abspath(__file__))

tracks = json.load(open(os.path.join(HERE, "tracks.json"), encoding="utf-8"))
segs = json.load(open(os.path.join(HERE, "segs.json"), encoding="utf-8"))
fps = 29.97

cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
panels = []
report = []

for si, segdata in enumerate(tracks["segs"]):
    si_real = segdata["start"]  # 段起点秒
    dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % list(tracks["segs"]).index(segdata)), encoding="utf-8")) if False else None
    # 找该段在 segs.json 里的索引
    seg_idx = next(i for i, s in enumerate(segs) if s[0] == segdata["start"] and s[1] == segdata["dur"])
    dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % seg_idx), encoding="utf-8"))
    det_by_t = {round(d["t"], 2): d for d in dets}
    tr = segdata["track"]

    # 一致性：track 点与其时刻最近的 YOLO 球距离
    dists, on = [], 0
    for p in tr:
        t = p["t"]
        best = None
        for tt, d in det_by_t.items():
            if abs(tt - t) < 0.051:
                best = d
                break
        if best and best["ball"]:
            dx = best["ball"]["x"] - p["x"]
            dy = best["ball"]["y"] - p["y"]
            dists.append((dx * dx + dy * dy) ** 0.5)
            on += 1
    if dists:
        report.append("seg %d [%ds] track=%d pts, 有球对齐 %d, 平均距离 %.3f, 最大 %.3f" %
                      (seg_idx, segdata["start"], len(tr), on, sum(dists) / len(dists), max(dists)))
    else:
        report.append("seg %d [%ds] track=%d pts, 无对齐球帧" % (seg_idx, segdata["start"], len(tr)))

    # 可视化 4 个时刻
    picks = [tr[int(len(tr) * f)] for f in (0.15, 0.4, 0.65, 0.9)]
    for p in picks:
        fno = int(round((segdata["start"] + p["t"]) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        cx, cy = int(p["x"] * W), int(p["y"] * H)
        # 放大示意框（zoom 倍率下的可见区域）
        z = p["zoom"] or 1.0
        vw, vh = int(W / z), int(H / z)
        cv2.rectangle(img, (cx - vw // 2, cy - vh // 2), (cx + vw // 2, cy + vh // 2), (255, 200, 60), 2)
        cv2.circle(img, (cx, cy), 8, (255, 200, 60), 2)
        cv2.putText(img, "focus z=%.2f" % z, (cx - vw // 2, cy - vh // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 60), 1)
        # YOLO 球框
        d = det_by_t.get(round(p["t"], 2))
        if d and d["ball"]:
            b = d["ball"]
            x1 = int((b["x"] - b["w"] / 2) * W); y1 = int((b["y"] - b["h"] / 2) * H)
            x2 = int((b["x"] + b["w"] / 2) * W); y2 = int((b["y"] + b["h"] / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, "seg%d t=%.1fs" % (seg_idx, p["t"]), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(img)
cap.release()

print("\n".join(report))
if panels:
    rows = [panels[i:i + 2] for i in range(0, len(panels), 2)]
    rows = [r + [panels[0]] * (2 - len(r)) for r in rows]
    cv2.imwrite(os.path.join(HERE, "tracks_vis.png"), np_vstack_rows(rows))
    print("saved tracks_vis.png")
