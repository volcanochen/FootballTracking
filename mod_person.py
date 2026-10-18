# -*- coding: utf-8 -*-
"""【模块 2】人感知：检测 → 跨帧关联（匈牙利 + IoU + 球衣颜色）→ 碎片合并 → 静态拒检

与球完全解耦：只依赖帧序列，不涉及任何球的信息。输出每条轨迹的归一化框与 ID。

自测/单独运行: python mod_person.py 2
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_W, SRC_H = 854, 356
FPS = 29.97

IMGSZ, CONF = 1280, 0.15            # 检测：COCO person 类，远景 35~52px 目标必须大分辨率
MIN_FRAMES, STATIC_PX = 8, 8.0      # 短轨迹丢弃；整段位移 <8px 判为静态误检（广告牌人物）
GATE_BASE, GATE_V, GATE_CAP = 18.0, 6.0, 140.0   # 关联门限（像素）
COLOR_W, COLOR_GATE = 0.9, 0.35     # 颜色代价权重 / 低 IoU 时的颜色接受门限
MISS_MAX = 60                       # 轨迹无匹配时的最大预测延续帧数

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def assign(cost_m, rows, cols):
    """全局最优分配（匈牙利优先）；无 scipy 时退化为按代价升序的贪心互近邻"""
    if not rows:
        return []
    rl, cl = sorted(set(rows)), sorted(set(cols))
    if HAS_SCIPY:
        ri = {v: i for i, v in enumerate(rl)}
        cim = {v: i for i, v in enumerate(cl)}
        M = np.full((len(rl), len(cl)), 1e6)
        for (a, b), c in cost_m.items():
            M[ri[a], cim[b]] = c
        r, c = linear_sum_assignment(M)
        return [(rl[a], cl[b]) for a, b in zip(r, c) if M[a, b] < 1e5]
    usedr, usedc, pairs = set(), set(), []
    for (a, b), _ in sorted(cost_m.items(), key=lambda kv: kv[1]):
        if a in usedr or b in usedc:
            continue
        usedr.add(a)
        usedc.add(b)
        pairs.append((a, b))
    return pairs


def color_hist(fr, box):
    """躯干上部球衣颜色：HSV 的 H 通道直方图（仅取有色像素）"""
    x1, y1, x2, y2 = [int(v) for v in box]
    y1b, y2b = y1 + int(0.15 * (y2 - y1)), y1 + int(0.55 * (y2 - y1))
    x1, x2 = max(0, x1), min(fr.shape[1], x2)
    y1b, y2b = max(0, y1b), min(fr.shape[0], y2b)
    if x2 - x1 < 3 or y2b - y1b < 3:
        return np.zeros(16), 0.0
    patch = cv2.cvtColor(fr[y1b:y2b, x1:x2], cv2.COLOR_BGR2HSV)
    s = patch[:, :, 1] / 255.0
    h = patch[:, :, 0].astype(np.float32) * 2.0
    w = (s > 0.25).astype(np.float32)
    if w.sum() < 5:
        return np.zeros(16), 0.0
    hist, _ = np.histogram(h, bins=16, range=(0, 360), weights=w)
    return hist / (hist.sum() + 1e-6), float(s.mean())


def color_dist(a, b):
    return 1.0 - float(np.minimum(a[0], b[0]).sum())


def detect(seg_id, frames):
    """逐帧检测并缓存（imgsz=1280, conf=0.15, COCO person）"""
    from ultralytics import YOLO
    cache = os.path.join(HERE, "dets_person_cache_s%d.npy" % seg_id)
    if os.path.exists(cache):
        return list(np.load(cache, allow_pickle=True))
    model = YOLO(os.path.join(HERE, "weights", "yolo26n.pt"))
    dets = []
    for i, fr in enumerate(frames):
        r = model.predict(fr, imgsz=IMGSZ, conf=CONF, classes=[0], device=0, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        dets.append(np.column_stack([b, c]) if len(b) else np.zeros((0, 5)))
        if i % 100 == 0:
            print("  det %d/%d n=%d" % (i, len(frames), len(b)), flush=True)
    np.save(cache, np.array(dets, dtype=object))
    return dets


def associate(dets, frames):
    """跨帧关联：恒速外推门限 + IoU + 球衣颜色直方图 ReID，匈牙利全局分配"""
    tracks, next_id = [], 1
    for f in range(len(dets)):
        fr = frames[f]
        cands_f = [{"box": d[:4], "conf": d[4], "ch": color_hist(fr, d[:4]),
                    "c": np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])}
                   for d in dets[f]]
        rows, cols, cost_m = [], [], {}
        for ti, t in enumerate(tracks):
            gap = f - t["lastf"]
            if gap > MISS_MAX:
                continue
            pred_c = t["c"] + t["v"] * gap
            pb = np.array([pred_c[0] - t["w"] / 2, pred_c[1] - t["h"] / 2,
                           pred_c[0] + t["w"] / 2, pred_c[1] + t["h"] / 2])
            gate = min(GATE_CAP, GATE_BASE + GATE_V * gap)
            for ci, cd in enumerate(cands_f):
                if np.linalg.norm(cd["c"] - pred_c) > gate:
                    continue
                db = cd["box"]
                ix1, iy1 = max(pb[0], db[0]), max(pb[1], db[1])
                ix2, iy2 = min(pb[2], db[2]), min(pb[3], db[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ar = max(1e-6, (pb[2] - pb[0]) * (pb[3] - pb[1]))
                br = max(1e-6, (db[2] - db[0]) * (db[3] - db[1]))
                iou = inter / (ar + br - inter)
                cdist = color_dist(t["ch"], cd["ch"])
                if not ((iou >= 0.45) or (iou >= 0.20 and cdist < COLOR_GATE)):
                    continue
                cost_m[(ti, ci)] = (1 - iou) + COLOR_W * cdist
                rows.append(ti)
                cols.append(ci)
        pairs = assign(cost_m, rows, cols)
        used = set()
        for ti, ci in pairs:
            t, cd = tracks[ti], cands_f[ci]
            used.add(ci)
            nv = 0.7 * t["v"] + 0.3 * (cd["c"] - t["prev_c"]) / max(1, f - t["lastf"])
            t["prev_c"], t["c"], t["v"] = t["c"], cd["c"], nv
            t["lastf"] = f
            t["w"] = cd["box"][2] - cd["box"][0]
            t["h"] = cd["box"][3] - cd["box"][1]
            t["ch"] = (0.8 * np.array(t["ch"][0]) + 0.2 * cd["ch"][0], t["ch"][1])
            t["boxes"][f] = tuple(cd["box"]) + (float(cd["conf"]),)
        for ci, cd in enumerate(cands_f):
            if ci in used or cd["conf"] < 0.25:
                continue
            tracks.append({"id": next_id, "boxes": {f: tuple(cd["box"]) + (float(cd["conf"]),)},
                           "c": cd["c"], "prev_c": cd["c"], "v": np.zeros(2), "lastf": f,
                           "w": cd["box"][2] - cd["box"][0], "h": cd["box"][3] - cd["box"][1],
                           "ch": cd["ch"]})
            next_id += 1
    return tracks


def merge_fragments(tracks):
    """ID 碎片合并：间歇漏检把同一个人拆成多段，按位置+颜色缝合"""
    tracks = [t for t in tracks if len(t["boxes"]) >= MIN_FRAMES]
    tracks.sort(key=lambda t: min(t["boxes"]))
    canon = []
    for t in tracks:
        best, bestc = None, 1e9
        for c in canon:
            gap = min(t["boxes"]) - max(c["boxes"])
            if gap < 1 or gap > 45:
                continue
            lb = c["boxes"][max(c["boxes"])]
            fb = t["boxes"][min(t["boxes"])]
            pc = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
            tc = np.array([(fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2])
            d = np.linalg.norm(tc - pc)
            if d > min(GATE_CAP, GATE_BASE + GATE_V * gap):
                continue
            cost = d / 100.0 + COLOR_W * color_dist(c["ch"], t["ch"])
            if cost < bestc:
                bestc, best = cost, c
        if best is not None:
            for f, b in t["boxes"].items():
                best["boxes"][f] = b
            best["ch"] = (0.5 * np.array(best["ch"][0]) + 0.5 * np.array(t["ch"][0]),
                          best["ch"][1])
        else:
            canon.append(t)
    return canon


def load_persons(seg_id, frames):
    """→ (persons, per_frame)。boxes 统一为**归一化**坐标（与球同一坐标系）"""
    dets = detect(seg_id, frames)
    persons = merge_fragments(associate(dets, frames))
    persons = [p for p in persons
               if np.array([np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                            for b in p["boxes"].values()]).std(axis=0).max() * SRC_W > STATIC_PX]
    for p in persons:
        p["boxes"] = {f: (b[0] / SRC_W, b[1] / SRC_H, b[2] / SRC_W, b[3] / SRC_H, b[4])
                      for f, b in p["boxes"].items()}
    per_frame = {}
    for p in persons:
        for f, b in p["boxes"].items():
            per_frame.setdefault(f, []).append((p["id"], b))
    return persons, per_frame


if __name__ == "__main__":
    si = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    start, dur, note = segs[si]
    f0, n = int(start * FPS), int(dur * FPS)
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    persons, per_frame = load_persons(si, frames)
    span = [max(p["boxes"]) - min(p["boxes"]) for p in persons]
    print("seg%d note=%s  persons=%d  中位跨度=%.0f帧  最长=%d帧" % (
        si, note, len(persons), np.median(span) if span else 0, max(span) if span else 0))
