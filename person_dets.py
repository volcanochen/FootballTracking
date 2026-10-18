# -*- coding: utf-8 -*-
"""批量生成人物检测缓存（dets_person_cache_s{si}.npy）

与 demo_person_reid.py 的检测段完全一致（yolo26n COCO person 类, imgsz=1280, conf=0.15），
只做检测与缓存，不做关联/实验——用于给 auto_director 的多个段落准备输入。

用法: python person_dets.py 2,3,6,10
"""
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
FPS = 29.97
IMGSZ, CONF = 1280, 0.15

segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
ids = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2").split(",")]
model = YOLO(os.path.join(HERE, "weights", "yolo26n.pt"))
cap = cv2.VideoCapture(VIDEO)

for si in ids:
    start, dur, note = segs[si]
    CACHE = os.path.join(HERE, "dets_person_cache_s%d.npy" % si)
    if os.path.exists(CACHE):
        arr = np.load(CACHE, allow_pickle=True)
        print("skip seg%d (cached, %d frames, note=%s)" % (si, len(arr), note), flush=True)
        continue
    f0, n = int(start * FPS), int(dur * FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    dets = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        r = model.predict(fr, imgsz=IMGSZ, conf=CONF, classes=[0], device=0, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        dets.append(np.column_stack([b, c]) if len(b) else np.zeros((0, 5)))
        if i % 100 == 0:
            print("  seg%d det %d/%d n=%d" % (si, i, n, len(b)), flush=True)
    np.save(CACHE, np.array(dets, dtype=object))
    print("saved %s (%d frames, note=%s)" % (os.path.basename(CACHE), len(dets), note), flush=True)
cap.release()
print("done")
