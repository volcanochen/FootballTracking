# -*- coding: utf-8 -*-
"""微调模型评测：runs/ball11n/weights/best.pt 重跑 8 段 → dets_ft + dets_mot_ft。
- ball：微调模型每帧最大分检出（不做单帧强求——低置信帧交给 Viterbi 补）
- people：沿用现有 dets（YOLO person 不变）
- ballMot：色度差分运动证据（与 motion_pass 同口径）
之后：VIT_DET=ft python viterbi.py → dets_vit（关键帧锚定 + 物理补轨迹）
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97

from ultralytics import YOLO
model = YOLO(os.path.join(HERE, "runs", "ball11n", "weights", "best.pt"))

os.makedirs(os.path.join(HERE, "dets_ft"), exist_ok=True)
os.makedirs(os.path.join(HERE, "dets_mot_ft"), exist_ok=True)
cap = cv2.VideoCapture(VIDEO)
for si in range(8):
    dp_ = os.path.join(HERE, "dets", "%d.json" % si)
    if not os.path.exists(dp_):
        continue
    old = json.load(open(dp_, encoding="utf-8"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
    prev_chr = None
    out = []
    mot_out = []
    for i in range(len(old)):
        ok, fr = cap.read()
        if ok:
            for _ in range(2):
                ok2, _f = cap.read()
                if not ok2:
                    break
        if not ok:
            out.append({"f": old[i]["f"], "t": old[i]["t"], "ball": None,
                        "ballAll": [], "people": old[i]["people"]})
            mot_out.append({"f": old[i]["f"], "t": old[i]["t"], "ballMot": None})
            continue
        chr_ = fr[:, :, 0].astype(np.float32) - fr[:, :, 2].astype(np.float32)
        H, W = fr.shape[:2]

        def mot_at(x, y):
            if prev_chr is None:
                return None
            px, py = int(x * W), int(y * H)
            if px < 4 or py < 4 or px >= W - 4 or py >= H - 4:
                return None
            return float(np.abs(chr_[py - 4:py + 5, px - 4:px + 5] - prev_chr[py - 4:py + 5, px - 4:px + 5]).mean())

        up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        r = model.predict(up, imgsz=1280, conf=0.10, verbose=False)[0]
        balls = []
        for b in r.boxes:
            cls_id = int(b.cls[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xywhn[0]]
            balls.append({"x": x1, "y": y1, "w": x2, "h": y2, "score": float(b.conf[0])})
        balls.sort(key=lambda d: -d["score"])
        ball = balls[0] if balls else None
        bm = None
        if ball:
            bm = mot_at(ball["x"], ball["y"])
        out.append({"f": old[i]["f"], "t": old[i]["t"], "ball": ball,
                    "ballAll": balls[:3], "people": old[i]["people"]})
        mot_out.append({"f": old[i]["f"], "t": old[i]["t"], "ballMot": bm})
        prev_chr = chr_
    with open(os.path.join(HERE, "dets_ft", "%d.json" % si), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False)
    with open(os.path.join(HERE, "dets_mot_ft", "%d.json" % si), "w", encoding="utf-8") as fp:
        json.dump(mot_out, fp, ensure_ascii=False)
    nb = sum(1 for r in out if r["ball"])
    hi = sum(1 for r in out if r["ball"] and r["ball"]["score"] > 0.5)
    print("seg %d ballFrames=%d/%d hiConf(>0.5)=%d" % (si, nb, len(out), hi), flush=True)
cap.release()
print("done")
