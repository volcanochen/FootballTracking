# -*- coding: utf-8 -*-
"""数据集 v2：真球=白色小球。
- 种子：dets.ballAll 中远离固定光源 (0.630,0.565) 的 YOLO 球检出
- 传播：±4 采样帧内最近的白色候选（white_candidates）继承标签（时间连续性）
- 负样本：无球帧（含光源区域作为背景 → 学会忽略光源）
- 按段划分：train 0,1,2,3,5,6 / val 4,7
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
LIGHT = (0.630, 0.565)
from make_sheets2 import white_candidates  # noqa: E402


# 1) 种子：来自 seeds_v2.json（YOLO score>=0.30 非光源非人框）。
#    剔除两类坏种：我评审的 7 个（手套/球鞋）+ 用户网页评审标错的 S 框
BAD_SEEDS = {29, 37, 57, 60, 62, 63, 64}
_sv = json.load(open(os.path.join(HERE, "seeds_v2.json"), encoding="utf-8"))
seeds = {}
white_by_frame = {}
from collections import Counter
for k, s in enumerate(_sv):
    if k in BAD_SEEDS:
        continue
    seeds.setdefault((s["seg"], s["i"]), []).append((s["x"], s["y"], 18.0))
print("seed frames=%d total=%d (bad excluded: %d)" % (
    len(seeds), sum(len(v) for v in seeds.values()), len(BAD_SEEDS)))

# 用户评审结果：坏框剔除；坏种子的衍生传播框（±4 帧内 P）一并剔除
_rr = json.load(open(os.path.join(HERE, "review_result.json"), encoding="utf-8"))
user_bad_S = set()      # (seg, i) 种子被标错
user_bad_box = set()    # "sX_iYYYY#S0" 原样
for b in _rr["badBoxes"]:
    fid, tag = b.split("#")
    user_bad_box.add(b)
    if tag.startswith("S"):
        si, i = int(fid.split("_")[0][1:]), int(fid.split("_i")[1])
        user_bad_S.add((si, i))
# 坏种子前后 ±4 帧的 P 全部视为可疑（由坏种子传播而来）
user_bad_P_frames = set()
for (bsi, bi) in user_bad_S:
    for di in range(bi - 4, bi + 5):
        user_bad_P_frames.add((bsi, di))
print("user bad seeds=%d suspicious P frames=%d" % (len(user_bad_S), len(user_bad_P_frames)))

# 2) 传播：需要逐帧白候选
cap = cv2.VideoCapture(VIDEO)
prop = {}
for si in range(8):
    if not os.path.exists(os.path.join(HERE, "dets", "%d.json" % si)):
        continue
    dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % si), encoding="utf-8"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
    fi_video = 0
    idx = 0
    while idx < len(dets):
        ok, fr = cap.read()
        if not ok:
            break
        fi_video += 1
        if fi_video - 1 != idx * 3:
            continue
        white_by_frame[(si, idx)] = white_candidates(fr)
        idx += 1
cap.release()

for (si, i0), balls in seeds.items():
    for (x0, y0, s0) in balls:
        for di in range(max(0, i0 - 4), min(len(json.load(open(os.path.join(HERE, "dets", "%d.json" % si)))) , i0 + 5)):
            if di == i0:
                continue
            best, bd = None, 0.03
            for c in white_by_frame.get((si, di), []):
                d = ((c["x"] - x0) ** 2 + (c["y"] - y0) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, c
            if best:
                prop.setdefault((si, di), []).append((best["x"], best["y"], max(14, 3.5 * best["area"] ** 0.5)))
print("propagated frames=%d total=%d" % (len(prop), sum(len(v) for v in prop.values())))

pos_by_frame = {}
for k in set(list(seeds.keys()) + list(prop.keys())):
    pos_by_frame[k] = seeds.get(k, []) + prop.get(k, [])

# 3) 导出（train segs 0,1,2,3,5,6 / val 4,7；负帧每 15 个采样帧取 1）
DS = os.path.join(HERE, "dataset2")
if os.path.exists(DS):
    shutil.rmtree(DS)
for split in ("train", "val"):
    os.makedirs(os.path.join(DS, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(DS, split, "images"), exist_ok=True)

NEG_EVERY = 15
need_pos = set(pos_by_frame.keys())
need_neg = set((si, i) for (si, i) in white_by_frame if (si, i) not in need_pos and i % NEG_EVERY == 0)
need = sorted(need_pos | need_neg)
by_seg = {}
for (si, i) in need:
    by_seg.setdefault(si, []).append(i)

cap = cv2.VideoCapture(VIDEO)
n_pos = n_neg = 0
for si in sorted(by_seg):
    split = "val" if si in (4, 7) else "train"
    dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % si), encoding="utf-8"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
    fi_video = 0
    idx = 0
    while idx < len(dets):
        ok, fr = cap.read()
        if not ok:
            break
        fi_video += 1
        if fi_video - 1 != idx * 3:
            continue
        if idx in by_seg[si]:
            up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            name = "s%d_i%04d" % (si, idx)
            cv2.imwrite(os.path.join(DS, split, "images", name + ".jpg"), up,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            # 按评审结果过滤：S/P 分别按序号对应；坏种子帧±4 内的 P 一律剔除
            s_boxes = seeds.get((si, idx), [])
            p_boxes = prop.get((si, idx), [])
            fid = "s%d_i%04d" % (si, idx)
            boxes = []
            for bi, (x, y, s854) in enumerate(s_boxes):
                if (si, idx) in user_bad_S or ("%s#S%d" % (fid, bi)) in user_bad_box:
                    continue
                boxes.append((x, y, s854))
            if (si, idx) not in user_bad_P_frames:
                for pi, (x, y, s854) in enumerate(p_boxes):
                    if ("%s#P%d" % (fid, pi)) in user_bad_box:
                        continue
                    boxes.append((x, y, s854))
            with open(os.path.join(DS, split, "labels", name + ".txt"), "w") as fp:
                for (x, y, s854) in boxes:
                    s = s854 * 2.0 / 1708.0
                    fp.write("0 %.4f %.4f %.4f %.4f\n" % (x, y, s, s))
            n_pos += 1 if boxes else 0
            n_neg += 0 if boxes else 1
        idx += 1
cap.release()
with open(os.path.join(DS, "ball.yaml"), "w") as fp:
    fp.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n" % DS.replace("\\", "/"))
print("images: pos=%d neg=%d -> %s" % (n_pos, n_neg, DS))
