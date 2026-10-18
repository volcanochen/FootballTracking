# -*- coding: utf-8 -*-
"""分析颜色候选的 BGR 通道特征：
真球（与 YOLO 球框重合的 cyan 候选） vs 疑似误检（无 YOLO 支持的候选）
→ 找 B-G 通道差的可分阈值。球是青色：蓝 > 绿；绿草/白线反光：绿 >= 蓝。
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from color_ball import cyan_candidates  # noqa: E402

VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

true_bg, true_gr, true_area = [], [], []
false_bg, false_gr = [], []
samples = []   # 保存一些误检样例用于目检

for si in range(len(segs)):
    if not os.path.exists(os.path.join(HERE, "dets", "%d.json" % si)):
        continue
    dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % si), encoding="utf-8"))
    start = segs[si][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    fi = 0
    for r in dets:
        ok, fr = cap.read()
        if not ok:
            break
        if r["f"] % 3 != 0:
            fi += 1
            continue
        cands = cyan_candidates(fr)
        yb = r["ball"]
        for c in cands:
            x = int(c["x"] * fr.shape[1]); y = int(c["y"] * fr.shape[0])
            patch = fr[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            b, g, rr = [float(v) for v in patch.reshape(-1, 3).mean(axis=0)]
            match = yb and abs(c["x"] - yb["x"]) < 0.05 and abs(c["y"] - yb["y"]) < 0.05
            if match:
                true_bg.append(b - g); true_gr.append(g - rr); true_area.append(c["area"])
            else:
                false_bg.append(b - g); false_gr.append(g - rr)
                if len(samples) < 40:
                    samples.append((si, fi, round(b - g, 1), round(g - rr, 1), c["area"], round(c["x"], 3), round(c["y"], 3)))
        fi += 1
    print("seg %d done" % si, flush=True)
cap.release()

import statistics as st
def stat(name, arr):
    if not arr:
        print(name, "empty"); return
    print("%s n=%d mean=%.1f median=%.1f p10=%.1f p90=%.1f min=%.1f max=%.1f" % (
        name, len(arr), st.mean(arr), st.median(arr), np.percentile(arr, 10), np.percentile(arr, 90), min(arr), max(arr)))

stat("TRUE  B-G", true_bg)
stat("TRUE  G-R", true_gr)
stat("FALSE B-G", false_bg)
stat("FALSE G-R", false_gr)
# 阈值扫描：B-G >= T 时保留，看真球保留率 / 误检剔除率
for T in (-10, -5, 0, 3, 5, 8, 10, 15):
    tr = sum(1 for v in true_bg if v >= T) / max(1, len(true_bg))
    fa = sum(1 for v in false_bg if v < T) / max(1, len(false_bg))
    print("T=%+d keepTrue=%.1f%% dropFalse=%.1f%%" % (T, 100 * tr, 100 * fa))
print("\nsample false cands (seg,f,B-G,G-R,area,x,y):")
for s in samples[:40]:
    print(s)
