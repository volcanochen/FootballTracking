# -*- coding: utf-8 -*-
"""人物身份保持（固定 ID）+ 遮挡推理验证

做三件事：
  1. 自写关联器：IoU + 恒速外推门限 + 球衣颜色直方图 ReID（离线可用未来帧）
  2. 遮挡补位：空洞 ≤OCC_FILL 帧按 smoothstep 插值补出位置（沿用项目 FILL_MAX 思想）
  3. 遮挡实验：人为抹掉某人连续 OCC_LEN 帧的检测（模拟被挡死），检验
       a) 该人被遮挡前后是否仍是同一 ID（身份保持）
       b) 遮挡期间推理出的位置 vs 被抹掉的真值框，误差多少像素 / IoU 多少

用法: python demo_person_reid.py [seg_id] [dur]
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
OCC_START, OCC_LEN = 150, 30       # 遮挡实验：从第 150 帧起抹掉 30 帧（约 1 秒）
OCC_FILL = 30                      # 补位最大空洞帧数（超过该值不硬猜位置，只保 ID）
GATE_BASE, GATE_V, GATE_CAP = 18.0, 6.0, 140.0
COLOR_W = 0.9                      # 颜色代价权重（相对 IoU 代价）
COLOR_GATE = 0.35                  # 低 IoU 时要求颜色足够接近才接受（防抢同色队友的框）
MISS_MAX = 60                      # 轨迹无匹配时的最大预测延续帧数
MIN_FRAMES, STATIC_PX = 8, 8.0

segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
si = int(sys.argv[1]) if len(sys.argv) > 1 else 2
start, dur, note = segs[si]
dur = min(dur, float(sys.argv[2]) if len(sys.argv) > 2 else 15.0)
f0, n = int(start * FPS), int(dur * FPS)
print("seg %d note=%s frames=%d" % (si, note, n), flush=True)

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
frames = []
for _ in range(n):
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()

model = YOLO(os.path.join(HERE, "weights", "yolo26n.pt"))
CACHE = os.path.join(HERE, "dets_person_cache_s%d.npy" % si)
if os.path.exists(CACHE):
    dets = np.load(CACHE, allow_pickle=True)
    print("load dets cache", dets.shape, flush=True)
else:
    dets = []
    for i, fr in enumerate(frames):
        r = model.predict(fr, imgsz=IMGSZ, conf=CONF, classes=[0], device=0, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        dets.append(np.column_stack([b, c]) if len(b) else np.zeros((0, 5)))
        if i % 100 == 0:
            print("  det %d/%d n=%d" % (i, len(frames), len(b)), flush=True)
    np.save(CACHE, np.array(dets, dtype=object))
print("dets ready", flush=True)


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
    hs = hist / (hist.sum() + 1e-6)
    return hs, float(s.mean())


def color_dist(a, b):
    return 1.0 - float(np.minimum(a[0], b[0]).sum())


def smoothstep(t):
    return t * t * (3 - 2 * t)


def associate(all_dets):
    """自写关联：全局最优分配（匈牙利）+ 运动门限 + 颜色 ReID + 宁缺勿错接受门限

    关键：匹配不上就宁可让该轨迹本帧留空（靠预测延续），不抢旁边队友的框——
    遮挡期间抢框会导致身份漂移（实测会漂到同色队友身上）。
    """
    try:
        from scipy.optimize import linear_sum_assignment
        has_sci = True
    except Exception:
        has_sci = False
    tracks = []
    next_id = 1
    for f in range(len(all_dets)):
        fr = frames[f]
        dets_f = all_dets[f]
        cands = [{"box": d[:4], "conf": d[4], "ch": color_hist(fr, d[:4]),
                  "c": np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])}
                 for d in dets_f]
        rows, cols, cost_m = [], [], {}
        for ti, t in enumerate(tracks):
            gap = f - t["lastf"]
            if gap > MISS_MAX:
                continue
            pred_c = t["c"] + t["v"] * gap
            pred_box = np.array([pred_c[0] - t["w"] / 2, pred_c[1] - t["h"] / 2,
                                 pred_c[0] + t["w"] / 2, pred_c[1] + t["h"] / 2])
            gate = min(GATE_CAP, GATE_BASE + GATE_V * gap)
            for ci, cd in enumerate(cands):
                if np.linalg.norm(cd["c"] - pred_c) > gate:
                    continue
                pb, db = pred_box, cd["box"]
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
        used_c = set()
        pairs = []
        if rows:
            if has_sci:
                M = np.full((len(set(rows)), len(set(cols))), 1e6)
                rl, cl = sorted(set(rows)), sorted(set(cols))
                ri = {v: i for i, v in enumerate(rl)}
                ci_ = {v: i for i, v in enumerate(cl)}
                for (a, b), c in cost_m.items():
                    M[ri[a], ci_[b]] = c
                r, c = linear_sum_assignment(M)
                for a, b in zip(r, c):
                    if M[a, b] < 1e5:
                        pairs.append((rl[a], cl[b]))
            else:
                for (a, b), c in sorted(cost_m.items(), key=lambda kv: kv[1]):
                    if a in [p[0] for p in pairs] or b in [p[1] for p in pairs]:
                        continue
                    pairs.append((a, b))
        for ti, ci in pairs:
            t, cd = tracks[ti], cands[ci]
            used_c.add(ci)
            nv = 0.7 * t["v"] + 0.3 * (cd["c"] - t["prev_c"]) / max(1, f - t["lastf"])
            t["prev_c"], t["c"], t["v"] = t["c"], cd["c"], nv
            t["lastf"] = f
            t["w"] = cd["box"][2] - cd["box"][0]
            t["h"] = cd["box"][3] - cd["box"][1]
            t["ch"] = (0.8 * np.array(t["ch"][0]) + 0.2 * cd["ch"][0], t["ch"][1])
            t["boxes"][f] = tuple(cd["box"]) + (float(cd["conf"]),)
        for ci, cd in enumerate(cands):
            if ci in used_c or cd["conf"] < 0.25:
                continue
            tracks.append({"id": next_id, "boxes": {f: tuple(cd["box"]) + (float(cd["conf"]),)},
                           "c": cd["c"], "prev_c": cd["c"], "v": np.zeros(2), "lastf": f,
                           "w": cd["box"][2] - cd["box"][0], "h": cd["box"][3] - cd["box"][1],
                           "ch": cd["ch"]})
            next_id += 1
    return tracks


def merge_fragments(tracks):
    """离线碎片合并：同一个人被拆成多段时，按运动 + 颜色接回同一 ID"""
    tracks = [t for t in tracks if len(t["boxes"]) >= MIN_FRAMES]
    tracks.sort(key=lambda t: min(t["boxes"]))
    canon = []
    for t in tracks:
        placed = False
        best, bestc = None, 1e9
        for c in canon:
            gap = min(t["boxes"]) - max(c["boxes"])
            if gap < 1 or gap > 45:
                continue
            pc = np.array([(c["boxes"][max(c["boxes"])][0] + c["boxes"][max(c["boxes"])][2]) / 2,
                           (c["boxes"][max(c["boxes"])][1] + c["boxes"][max(c["boxes"])][3]) / 2])
            tc = np.array([(t["boxes"][min(t["boxes"])][0] + t["boxes"][min(t["boxes"])][2]) / 2,
                           (t["boxes"][min(t["boxes"])][1] + t["boxes"][min(t["boxes"])][3]) / 2])
            dist = np.linalg.norm(tc - pc)
            if dist > min(GATE_CAP, GATE_BASE + GATE_V * gap):
                continue
            cost = dist / 100.0 + COLOR_W * color_dist(c["ch"], t["ch"])
            if cost < bestc:
                bestc, best = cost, c
        if best is not None:
            for f, b in t["boxes"].items():
                best["boxes"][f] = b
            best["ch"] = (0.5 * np.array(best["ch"][0]) + 0.5 * np.array(t["ch"][0]),
                          best["ch"][1])
            placed = True
        if not placed:
            canon.append(t)
    return canon


def fill_gaps(tracks):
    """遮挡补位：空洞 ≤OCC_FILL 帧按 smoothstep 插值补出位置"""
    filled = {}
    for t in tracks:
        fs = sorted(t["boxes"])
        for a, b in zip(fs, fs[1:]):
            if b - a - 1 == 0 or b - a - 1 > OCC_FILL:
                continue
            ba, bb = np.array(t["boxes"][a][:4]), np.array(t["boxes"][b][:4])
            for k in range(1, b - a):
                tt = smoothstep(k / (b - a))
                box = ba + (bb - ba) * tt
                t["boxes"][a + k] = tuple(box) + (0.0,)
                filled.setdefault(t["id"], []).append(a + k)
    return filled


# ---------- 基线：完整检测 ----------
base = merge_fragments(associate(dets))
print("[baseline] tracks=%d" % len(base), flush=True)

# ---------- 遮挡实验 ----------
target = max(base, key=lambda t: len(t["boxes"]))
tfs = sorted(target["boxes"])
occ_range = [f for f in tfs if OCC_START <= f < OCC_START + OCC_LEN]
gt = {f: target["boxes"][f] for f in occ_range}
print("[occlusion test] target id=%d frames=%d, hide %d frames (%d~%d) = %.1fs" % (
    target["id"], len(tfs), len(occ_range), OCC_START, OCC_START + OCC_LEN,
    len(occ_range) / FPS), flush=True)

dets_occ = []
for f in range(len(dets)):
    if f in gt:
        keep = []
        for d in dets[f]:
            cx = (d[0] + d[2]) / 2
            cy = (d[1] + d[3]) / 2
            gb = gt[f]
            gc = ((gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2)
            if abs(cx - gc[0]) < (gb[2] - gb[0]) / 2 and abs(cy - gc[1]) < (gb[3] - gb[1]) / 2:
                continue
            keep.append(d)
        dets_occ.append(np.array(keep, dtype=np.float32) if keep else np.zeros((0, 5)))
    else:
        dets_occ.append(dets[f])

occ_tracks = merge_fragments(associate(dets_occ))
occ_filled = fill_gaps(occ_tracks)

# 找遮挡后目标对应的 track（按颜色 + 位置匹配）
def match_target(tracks, ref, before=OCC_START):
    """用遮挡前（f<before）的重合框认出目标轨迹，避免按颜色误配到同队队友"""
    best, bestn = None, 0
    for t in tracks:
        n = 0
        for f, rb in ref["boxes"].items():
            if f >= before or f not in t["boxes"]:
                continue
            tb = t["boxes"][f]
            ix1, iy1 = max(tb[0], rb[0]), max(tb[1], rb[1])
            ix2, iy2 = min(tb[2], rb[2]), min(tb[3], rb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ar = (tb[2] - tb[0]) * (tb[3] - tb[1])
            br = (rb[2] - rb[0]) * (rb[3] - rb[1])
            if inter / max(1e-6, ar + br - inter) > 0.5:
                n += 1
        if n > bestn:
            bestn, best = n, t
    return best, bestn

recovered, nmatch = match_target(occ_tracks, target)
same_id_after = recovered is not None and (
    min(recovered["boxes"]) <= OCC_START and max(recovered["boxes"]) >= OCC_START + OCC_LEN)
print("[identity] recovered track: %s (matched %d pre-occlusion frames) | spans across occlusion: %s" % (
    "YES" if recovered is not None else "NO", nmatch, "YES" if same_id_after else "NO"), flush=True)
if recovered is not None:
    print("           len=%d  span=%d~%d" % (
        len(recovered["boxes"]), min(recovered["boxes"]), max(recovered["boxes"])), flush=True)

errs, ious, covered = [], [], 0
if recovered is not None:
    for f, gb in gt.items():
        if f in recovered["boxes"]:
            covered += 1
            pb = recovered["boxes"][f][:4]
            ec = ((pb[0] + pb[2]) / 2 - (gb[0] + gb[2]) / 2,
                  (pb[1] + pb[3]) / 2 - (gb[1] + gb[3]) / 2)
            errs.append(np.linalg.norm(ec))
            ix1, iy1 = max(pb[0], gb[0]), max(pb[1], gb[1])
            ix2, iy2 = min(pb[2], gb[2]), min(pb[3], gb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ar = (pb[2] - pb[0]) * (pb[3] - pb[1])
            br = (gb[2] - gb[0]) * (gb[3] - gb[1])
            ious.append(inter / max(1e-6, ar + br - inter))
if errs:
    print("[position inference] covered %d/%d occluded frames (%.0f%%)" % (
        covered, len(gt), 100.0 * covered / len(gt)), flush=True)
    print("           OFFLINE interp : center err mean=%.1fpx med=%.1fpx p90=%.1fpx | IoU mean=%.2f" % (
        np.mean(errs), np.median(errs), np.percentile(errs, 90), np.mean(ious)), flush=True)

if recovered is not None:
    pre = sorted([f for f in recovered["boxes"] if f < OCC_START])
    if len(pre) >= 6:
        a, b = recovered["boxes"][pre[-5]], recovered["boxes"][pre[-1]]
        v = (np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) -
             np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])) / (pre[-1] - pre[-5])
        last = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        cerr, ciou = [], []
        for f in sorted(gt):
            p = last + v * (f - pre[-1])
            g = gt[f]
            gc = np.array([(g[0] + g[2]) / 2, (g[1] + g[3]) / 2])
            cerr.append(np.linalg.norm(p - gc))
            pb = [p[0] - (b[2] - b[0]) / 2, p[1] - (b[3] - b[1]) / 2,
                  p[0] + (b[2] - b[0]) / 2, p[1] + (b[3] - b[1]) / 2]
            ix1, iy1 = max(pb[0], g[0]), max(pb[1], g[1])
            ix2, iy2 = min(pb[2], g[2]), min(pb[3], g[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ar = (pb[2] - pb[0]) * (pb[3] - pb[1])
            br = (g[2] - g[0]) * (g[3] - g[1])
            ciou.append(inter / max(1e-6, ar + br - inter))
        print("           CAUSAL extrap  : center err mean=%.1fpx med=%.1fpx p90=%.1fpx | IoU mean=%.2f" % (
            np.mean(cerr), np.median(cerr), np.percentile(cerr, 90), np.mean(ciou)), flush=True)
        print("           (person box height ~%.0fpx, frame 854x356)" % (b[3] - b[1]), flush=True)

# ---------- 输出视频 ----------
rng = np.random.default_rng(3)
palette = {t["id"]: tuple(int(x) for x in rng.integers(40, 240, 3)) for t in occ_tracks}
vw = cv2.VideoWriter(os.path.join(HERE, "demo_reid_s%d.mp4" % si),
                     cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                     (frames[0].shape[1], frames[0].shape[0]))
for f, fr in enumerate(frames):
    img = fr.copy()
    if OCC_START <= f < OCC_START + OCC_LEN and f in gt:
        gb = gt[f]
        cv2.rectangle(img, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])),
                      (0, 0, 255), 1)
        cv2.putText(img, "hidden truth", (int(gb[0]), int(gb[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    for t in occ_tracks:
        if f not in t["boxes"]:
            continue
        b = t["boxes"][f]
        is_pred = f in occ_filled.get(t["id"], [])
        col = (0, 140, 255) if is_pred else palette[t["id"]]
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col,
                      1 if is_pred else 2, cv2.LINE_4 if is_pred else cv2.LINE_8)
        cv2.putText(img, "%s%d" % ("pred " if is_pred else "", t["id"]),
                    (int(b[0]), max(11, int(b[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(img, "f=%d  id-kept=%s  occlusion %d~%d" % (
        f, "YES" if same_id_after else "NO", OCC_START, OCC_START + OCC_LEN),
        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "orange dashed = inferred during occlusion | red = hidden ground truth",
                (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    vw.write(img)
vw.release()
print("saved demo_reid_s%d.mp4" % si, flush=True)
