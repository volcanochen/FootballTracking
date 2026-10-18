# -*- coding: utf-8 -*-
"""构建 YOLO 微调数据集：
- 正样本：人工标注的 ball 贴片 + 时间连续传播（±4 采样帧内在 0.035 邻域的候选同判为球）
- 负样本：无任何候选的采样帧（全帧背景）
- 图像：2x 放大帧（1708x712），训练 imgsz=1280
- 划分：train = seg 0,1,2,3,5,6；val = seg 4,7（按段划分防时间泄漏）
"""
import json
import os
import shutil

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
from labels_manual import BALLS  # noqa: E402

sheets_meta = json.load(open(os.path.join(HERE, "sheets", "meta.json"), encoding="utf-8"))
tiles = sheets_meta["tiles"]

# 1) 种子正样本 → 时间传播
seed_pos = [(int(t), tiles[t]) for t in map(str, BALLS)]
pos_by_frame = {}   # (seg, i) -> [(x, y, area, src)]
for tid, m in seed_pos:
    pos_by_frame.setdefault((m["seg"], m["i"]), []).append((m["x"], m["y"], m["area"], m["src"]))

props = 0
for tid, m in seed_pos:
    si, i0 = m["seg"], m["i"]
    dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % si), encoding="utf-8"))
    cyans = json.load(open(os.path.join(HERE, "dets_cyan", "%d.json" % si), encoding="utf-8"))
    for di in range(max(0, i0 - 4), min(len(dets), i0 + 5)):
        if di == i0:
            continue
        # 预测位置=种子位置（0.4s 内球位移小，邻域放宽到 0.035）
        for c in cyans[di]["cands"]:
            if abs(c["x"] - m["x"]) < 0.035 and abs(c["y"] - m["y"]) < 0.035:
                pos_by_frame.setdefault((si, di), []).append((c["x"], c["y"], c["area"], "prop"))
                props += 1
                break
        for b in (dets[di].get("ballAll") or []):
            if abs(b["x"] - m["x"]) < 0.035 and abs(b["y"] - m["y"]) < 0.035:
                pos_by_frame.setdefault((si, di), []).append((b["x"], b["y"], max(6, (b["w"] * 854 * b["h"] * 356) ** 0.5), "prop"))
                props += 1
                break
print("seed=%d propagated=%d frames_with_ball=%d" % (len(seed_pos), props, len(pos_by_frame)))

# 2) 全帧帧索引：需要哪些帧（有球帧 + 负帧）
neg_frames = []
for f in sheets_meta["frames"]:
    key = (f["seg"], f["i"])
    if f["ncand"] == 0 and key not in pos_by_frame:
        neg_frames.append(key)
print("neg frames=%d" % len(neg_frames))

need = sorted(set(list(pos_by_frame.keys()) + neg_frames))
by_seg = {}
for (si, i) in need:
    by_seg.setdefault(si, []).append(i)

# 3) 导出图像 + 标签
DS = os.path.join(HERE, "dataset")
if os.path.exists(DS):
    shutil.rmtree(DS)
for split in ("train", "val"):
    os.makedirs(os.path.join(DS, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(DS, split, "images"), exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
n_img = 0
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
            H, W = up.shape[:2]
            boxes = []
            for (x, y, area, src) in pos_by_frame.get((si, idx), []):
                side_px = max(14, 3.5 * (area ** 0.5)) * 2.0 / W   # 归一化边长
                boxes.append((x, y, side_px))
            name = "s%d_i%04d" % (si, idx)
            cv2.imwrite(os.path.join(DS, split, "images", name + ".jpg"), up,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(DS, split, "labels", name + ".txt"), "w") as fp:
                for (x, y, s) in boxes:
                    fp.write("0 %.4f %.4f %.4f %.4f\n" % (x, y, s, s))
            n_img += 1
        idx += 1
cap.release()
with open(os.path.join(DS, "ball.yaml"), "w") as fp:
    fp.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n" % DS.replace("\\", "/"))
print("images written=%d -> %s" % (n_img, DS))
