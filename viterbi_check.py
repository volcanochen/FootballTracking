# -*- coding: utf-8 -*-
"""Viterbi 轨迹验证：
1) 量化：viterbi 球与 YOLO 球（同帧）的一致率/平均距离
2) 可视化：原帧 + 青色候选(绿点) + YOLO球框(橙) + viterbi选球(红圈)
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30


def np_vstack_rows(rows):
    return np.vstack([np.hstack(r) for r in rows])


panels = []
report = []
for si in range(len(segs)):
    dp_ = os.path.join(HERE, "dets", "%d.json" % si)
    vp = os.path.join(HERE, "dets_vit", "%d.json" % si)
    cp = os.path.join(HERE, "dets_cyan", "%d.json" % si)
    if not (os.path.exists(dp_) and os.path.exists(vp)):
        continue
    dets = json.load(open(dp_, encoding="utf-8"))
    vit = json.load(open(vp, encoding="utf-8"))
    cyans = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None

    # 量化一致性
    agree = n = 0
    dists = []
    for i, r in enumerate(dets):
        yb = r.get("ball")
        vb = vit[i].get("ball")
        if not yb or not vb:
            continue
        n += 1
        d = ((yb["x"] - vb["x"]) ** 2 + (yb["y"] - vb["y"]) ** 2) ** 0.5
        dists.append(d)
        if d < 0.05:
            agree += 1
    report.append("seg %d [%3ds] yolo有球&vit有球=%d  一致(<0.05)=%.0f%%  平均距=%.3f  最大=%.3f" % (
        si, segs[si][0], n, 100 * agree / max(1, n),
        sum(dists) / max(1, len(dists)), max(dists) if dists else 0))

    # 可视化 4 个采样时刻
    start = segs[si][0]
    for fr_ in (0.15, 0.4, 0.65, 0.9):
        idx = int(fr_ * len(vit))
        r, v = dets[idx], vit[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + r["t"]) * fps))
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        if cyans and idx < len(cyans):
            for c in cyans[idx]["cands"]:
                cv2.circle(img, (int(c["x"] * W), int(c["y"] * H)), 5, (0, 255, 0), 1)
        yb = r.get("ball")
        if yb:
            cv2.rectangle(img, (int((yb["x"] - 0.012) * W), int((yb["y"] - 0.012) * H)),
                          (int((yb["x"] + 0.012) * W), int((yb["y"] + 0.012) * H)), (0, 165, 255), 1)
        vb = v.get("ball")
        if vb:
            cv2.circle(img, (int(vb["x"] * W), int(vb["y"] * H)), 9, (0, 0, 255), 2)
            tag = "V" if v.get("real") else "i"
            cv2.putText(img, tag, (int(vb["x"] * W) + 10, int(vb["y"] * H) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, "seg%d t=%.1fs" % (si, r["t"]), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(img)
cap.release()

print("\n".join(report))
if panels:
    rows = [panels[i:i + 2] for i in range(0, len(panels), 2)]
    while rows and len(rows[-1]) < 2:
        rows[-1].append(panels[0])
    cv2.imwrite(os.path.join(HERE, "viterbi_vis.png"), np_vstack_rows(rows))
    print("saved viterbi_vis.png")
