# -*- coding: utf-8 -*-
"""把 Viterbi 轨迹标注成视频：每帧画 选球圈 + 轨迹尾巴。
红圈=真实检测(V) 橙圈=插值(i)，30fps 平滑展开（10fps 锚点间线性插值）。
输出 anno_s<i>.mp4（无音频），供逐段播放审看。"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.environ.get("TRAJ_DIR", "dets_track")   # 轨迹源目录（dets_track=锚点局部跟踪）
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))

for si in range(8):
    vp = os.path.join(HERE, TRAJ_DIR, "%d.json" % si)
    if not os.path.exists(vp):
        continue
    vit = json.load(open(vp, encoding="utf-8"))
    start, dur = segs[si][0], segs[si][1]
    n_dets = len(vit)                       # 10fps 锚点
    n_frames = int(dur * FPS)
    # 锚点数组：ball 可能为 None → 前后最近锚点线性插值填充（>6 锚点距离则置 None）
    ax = np.full(n_dets, np.nan)
    ay = np.full(n_dets, np.nan)
    areal = np.zeros(n_dets, dtype=bool)
    asrc = [None] * n_dets
    for i, v in enumerate(vit):
        b = v.get("ball")
        if b:
            ax[i], ay[i], areal[i] = b["x"], b["y"], bool(v.get("real"))
            asrc[i] = v.get("src")
    # 填 None 锚点（线性插值锚点之间；长空洞保持 nan → 不画）
    for i in range(n_dets):
        if not np.isnan(ax[i]):
            continue
        j = i - 1
        while j >= 0 and np.isnan(ax[j]):
            j -= 1
        k = i + 1
        while k < n_dets and np.isnan(ax[k]):
            k += 1
        if j >= 0 and k < n_dets and k - j <= 6:
            u = (i - j) / (k - j)
            ax[i] = ax[j] + (ax[k] - ax[j]) * u
            ay[i] = ay[j] + (ay[k] - ay[j]) * u

    out_path = os.path.join(HERE, "anno_s%d.mp4" % si)
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (854, 356))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * FPS))
    trail = []
    for vf in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        # 该视频帧对应的 10fps 锚点下标（线性取值）
        pos = vf / FPS / 0.1001   # dets 采样周期≈0.1001s
        i0 = int(pos)
        if i0 >= n_dets - 1:
            i0 = n_dets - 2
        u = pos - i0
        if 0 <= i0 < n_dets and not np.isnan(ax[i0]) and not np.isnan(ax[min(i0 + 1, n_dets - 1)]):
            x = ax[i0] + (ax[i0 + 1] - ax[i0]) * u
            y = ay[i0] + (ay[i0 + 1] - ay[i0]) * u
            # 真实性：取最近的非 nan 锚点
            ii = i0 if not np.isnan(ax[i0]) else i0 + 1
            real = areal[ii]
            src = asrc[ii]
            px, py = int(x * 854), int(y * 356)
            trail.append((px, py, real))
            if len(trail) > 18:
                trail.pop(0)
            for ti, (tx, ty, tr) in enumerate(trail[:-1]):
                cv2.circle(fr, (tx, ty), 2, (0, 200, 120) if tr else (120, 160, 255), -1)
            if src == "A":
                cv2.circle(fr, (px, py), 10, (0, 220, 0), 2)
                cv2.putText(fr, "A", (px + 11, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            elif src == "w":
                cv2.circle(fr, (px, py), 9, (0, 230, 230), 2)
                cv2.putText(fr, "W", (px + 10, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 230), 1)
            elif real:
                cv2.circle(fr, (px, py), 10, (0, 0, 255), 2)
                cv2.putText(fr, "V", (px + 11, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            else:
                cv2.circle(fr, (px, py), 10, (0, 150, 255), 2)
                cv2.putText(fr, "i", (px + 11, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1)
        cv2.putText(fr, "seg%d %.1fs" % (si, vf / FPS), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        wr.write(fr)
    wr.release()
    print("saved %s (%.1fs)" % (out_path, n_frames / FPS), flush=True)
cap.release()
print("done")
