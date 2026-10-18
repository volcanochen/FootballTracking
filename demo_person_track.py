# -*- coding: utf-8 -*-
"""人物检测+跟踪演示（对应方案 A 管线，含离线 ID 一致化修正）

取 segs_cur.json 一段（>10s）：
  YOLO26n@imgsz1280 逐帧检测 → ByteTrack 关联 → 离线修正
    ① 重复合并（时间重叠 + 高 IoU 视为同一人被拆成两个 ID）
    ② 碎片合并（断链两端按速度外推门限接合，对应方案 A ④b）
    ③ 静态拒检（整段中心位移过小 → 广告牌海报/看台观众，画灰虚线）

用法: python demo_person_track.py [seg_id] [max_dur]
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
IMGSZ = 1280
CONF = 0.15
STATIC_PX = 8.0
MIN_FRAMES = 8
MERGE_GAP = 45          # 碎片合并最大断链帧数
MERGE_BASE = 18.0       # 门限基础（像素）
MERGE_V = 6.0           # 门限速度项（像素/帧）
MERGE_CAP = 140.0
DUP_IOU = 0.55          # 重复判定 IoU

segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
si = int(sys.argv[1]) if len(sys.argv) > 1 else 2
max_dur = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
start, dur, note = segs[si]
dur = min(dur, max_dur)
f0 = int(start * FPS)
n = int(dur * FPS)
print("seg %d [%ds,%ds] note=%s frames=%d" % (si, start, start + dur, note, n), flush=True)

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
frames = []
for _ in range(n):
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
print("read %d frames %s" % (len(frames), frames[0].shape), flush=True)

model = YOLO(os.path.join(HERE, "weights", "yolo26n.pt"))
raw = []
for i, fr in enumerate(frames):
    r = model.track(fr, imgsz=IMGSZ, conf=CONF, iou=0.6, classes=[0], device=0,
                    persist=True, tracker=os.path.join(HERE, "bytetrack_person.yaml"),
                    verbose=False)[0]
    boxes = []
    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.cpu().numpy()
        xy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, t in zip(xy, cf, ids):
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(c), int(t)))
    raw.append({"f": i, "boxes": boxes})
    if i % 100 == 0:
        print("  tracked %d/%d boxes=%d" % (i, len(frames), len(boxes)), flush=True)

by_id = {}
for rec in raw:
    for x1, y1, x2, y2, c, t in rec["boxes"]:
        by_id.setdefault(t, []).append((rec["f"], x1, y1, x2, y2, c))

raw_tracks = {t: sorted(v) for t, v in by_id.items() if len(v) >= MIN_FRAMES}
print("raw tracks=%d  lens=%s" % (
    len(raw_tracks), sorted(len(v) for v in raw_tracks.values())), flush=True)


def box_of(it):
    f, x1, y1, x2, y2, c = it
    return np.array([x1, y1, x2, y2])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ar = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    br = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (ar + br - inter)


def center(it):
    return np.array([(it[1] + it[3]) / 2, (it[2] + it[4]) / 2])


# ---- ① 重复合并：时间重叠且 IoU 高 ----
groups = [[t, list(v)] for t, v in raw_tracks.items()]
merged_dup = True
while merged_dup:
    merged_dup = False
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            fa = {it[0]: it for it in groups[a][1]}
            fb = {it[0]: it for it in groups[b][1]}
            common = set(fa) & set(fb)
            if len(common) < 3:
                continue
            ious = [iou(box_of(fa[f]), box_of(fb[f])) for f in common]
            if np.median(ious) >= DUP_IOU:
                keep = {}
                for it in groups[a][1] + groups[b][1]:
                    if it[0] not in keep or it[5] > keep[it[0]][5]:
                        keep[it[0]] = it
                groups[a][1] = sorted(keep.values())
                groups.pop(b)
                merged_dup = True
                break
        if merged_dup:
            break
print("after dup-merge: %d tracks" % len(groups), flush=True)


# ---- ② 碎片合并：断链两端速度外推门限接合 ----
def vel(items, k=5):
    if len(items) < 2:
        return np.zeros(2)
    a, b = items[-k], items[-1]
    dt = max(1, b[0] - a[0])
    return (center(b) - center(a)) / dt


canon = []
for _, items in sorted(groups, key=lambda g: g[1][0][0]):
    placed = False
    for c in canon:
        gap = items[0][0] - c[-1][0]
        if gap < 1 or gap > MERGE_GAP:
            continue
        v = vel(c)
        pred = center(c[-1]) + v * gap
        gate = min(MERGE_CAP, MERGE_BASE + MERGE_V * gap)
        if np.linalg.norm(center(items[0]) - pred) <= gate:
            c.extend(items)
            placed = True
            break
    if not placed:
        canon.append(list(items))
print("after fragment-merge: %d tracks  lens=%s" % (
    len(canon), sorted(len(c) for c in canon)), flush=True)


# ---- ③ 静态拒检 ----
alive, static = {}, {}
for idx, items in enumerate(canon, start=1):
    cs = np.array([center(it) for it in items])
    disp = np.linalg.norm(cs - cs.mean(axis=0), axis=1).max()
    (static if disp < STATIC_PX else alive)[idx] = items
print("FINAL alive=%d static_rejected=%d" % (len(alive), len(static)), flush=True)
for t, items in sorted(alive.items()):
    print("   id%-3d frames=%3d  span=%d~%d  disp=%.0fpx" % (
        t, len(items), items[0][0], items[-1][0],
        np.linalg.norm(np.array([center(i) for i in items]) -
                       np.array([center(i) for i in items]).mean(axis=0), axis=1).max()), flush=True)
for t, items in sorted(static.items()):
    print("   REJ id%-3d frames=%3d  disp=%.0fpx (static)" % (
        t, len(items),
        np.linalg.norm(np.array([center(i) for i in items]) -
                       np.array([center(i) for i in items]).mean(axis=0), axis=1).max()), flush=True)

# ---- 绘图 ----
rng = np.random.default_rng(7)
palette = {t: tuple(int(x) for x in rng.integers(40, 240, 3)) for t in alive}
per_alive, per_static = {}, {}
for t, items in alive.items():
    for it in items:
        per_alive.setdefault(it[0], []).append((t, it[1], it[2], it[3], it[4], it[5]))
for t, items in static.items():
    for it in items:
        per_static.setdefault(it[0], []).append((t, it[1], it[2], it[3], it[4], it[5]))

vw = cv2.VideoWriter(os.path.join(HERE, "demo_person_s%d.mp4" % si),
                     cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                     (frames[0].shape[1], frames[0].shape[0]))
for i, fr in enumerate(frames):
    img = fr.copy()
    for (t, x1, y1, x2, y2, c) in per_static.get(i, []):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (150, 150, 150), 1, cv2.LINE_4)
        cv2.putText(img, "rej", (int(x1), max(9, int(y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
    for (t, x1, y1, x2, y2, c) in per_alive.get(i, []):
        col = palette[t]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, "%d" % t, (int(x1), max(11, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        for p in [q for q in alive[t] if i - 18 <= q[0] <= i]:
            cv2.circle(img, (int((p[1] + p[3]) / 2), int((p[2] + p[4]) / 2)), 2, col, -1)
        cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, (0, 0, 255), -1)
    cv2.putText(img, "f=%d t=%.1fs  people=%d  rejected=%d" %
                (i, i / FPS, len(per_alive.get(i, [])), len(per_static.get(i, []))),
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "yolo26n@1280 + bytetrack + offline id-merge | gray dashed = static rejected",
                (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    vw.write(img)
vw.release()
print("saved demo_person_s%d.mp4" % si, flush=True)
