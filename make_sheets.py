# -*- coding: utf-8 -*-
"""标注触表生成：为多模态大模型标注准备素材。

策略（时间连续 + 多样性兼顾）：
  - 8 个测试段中每隔 SKIP 个采样帧取 1 帧（dets 已有 10fps 记录）
  - 每帧生成两类素材：
    1) 全帧缩略图（标记所有候选位置）→ 用来发现"没有任何候选的真球"（漏检）
    2) 每个候选的放大贴片（含 YOLO ballAll 全部球检出）→ 逐个判定 ball/非ball
  - 输出 sheets/tile_<n>.png（40 片/张，带全局 ID）+ sheets/full_<n>.png（6 帧/张）
  - meta.json 记录每个 tile 的全局 ID → (seg, frameIdx, x, y, area, src)
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
SKIP = 12          # 每 12 个采样帧（1.2s）取 1 帧 → 8 段约 78 帧
TILE = 24          # 贴片半径（原分辨率 px），贴片 48x48 → 放大 4x = 192
SHEET_COLS = 8

os.makedirs(os.path.join(HERE, "sheets"), exist_ok=True)
cap = cv2.VideoCapture(VIDEO)

tiles = []      # (img, meta)
fulls = []      # (img, meta)
meta = {}
tid = 0
fid = 0
for si in range(8):
    dp_ = os.path.join(HERE, "dets", "%d.json" % si)
    if not os.path.exists(dp_):
        continue
    dets = json.load(open(dp_, encoding="utf-8"))
    cyans = json.load(open(os.path.join(HERE, "dets_cyan", "%d.json" % si), encoding="utf-8"))
    mots = json.load(open(os.path.join(HERE, "dets_mot", "%d.json" % si), encoding="utf-8"))
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
        # idx 是 dets 记录号
        if idx % SKIP == 0:
            H, W = fr.shape[:2]
            cands = []
            for c in cyans[idx]["cands"]:
                cands.append((c["x"], c["y"], c["area"], "cyan"))
            for b in (dets[idx].get("ballAll") or []):
                cands.append((b["x"], b["y"], int(max(4, b["w"] * W * b["h"] * H) ** 0.5), "yolo"))
            # 去重（距离 <0.02 视为同点）
            dedup = []
            for c in cands:
                dup = False
                for d in dedup:
                    if abs(c[0] - d[0]) < 0.02 and abs(c[1] - d[1]) < 0.02:
                        dup = True
                        break
                if not dup:
                    dedup.append(c)
            fulls.append((fr.copy(), (si, idx, dedup, dets[idx]["t"])))
            for (x, y, area, src) in dedup:
                px, py = int(x * W), int(y * H)
                if not (TILE < px < W - TILE and TILE < py < H - TILE):
                    continue
                tile = cv2.resize(fr[py - TILE:py + TILE, px - TILE:px + TILE], (192, 192),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile, "#%d" % tid, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                tiles.append(tile)
                meta[str(tid)] = {"seg": si, "i": idx, "t": dets[idx]["t"], "x": round(x, 4),
                                  "y": round(y, 4), "area": int(area), "src": src}
                tid += 1
        idx += 1
cap.release()

# 拼触表
for i in range(0, len(tiles), SHEET_COLS):
    row = tiles[i:i + SHEET_COLS]
    while len(row) < SHEET_COLS:
        row.append(np.zeros((192, 192, 3), np.uint8))
    sheet = np.hstack(row)
    cv2.imwrite(os.path.join(HERE, "sheets", "tile_%02d.png" % (i // SHEET_COLS)), sheet)

FULL_COLS = 3
for i in range(0, len(fulls), FULL_COLS):
    row = []
    for j in range(i, min(i + FULL_COLS, len(fulls))):
        img, (si, idx, cands, tsec) = fulls[j]
        H, W = img.shape[:2]
        vis = cv2.resize(img, (1284, 534))
        sx, sy = 1284 / W, 534 / H
        for k, (x, y, area, src) in enumerate(cands):
            cv2.circle(vis, (int(x * sx), int(y * sy)), 9, (0, 0, 255), 2)
            cv2.putText(vis, str(k), (int(x * sx) + 10, int(y * sy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, "seg%d i=%d t=%.1f" % (si, idx, tsec), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        row.append(vis)
    while len(row) < FULL_COLS:
        row.append(np.zeros((534, 1284, 3), np.uint8))
    cv2.imwrite(os.path.join(HERE, "sheets", "full_%02d.png" % (i // FULL_COLS)), np.hstack(row))

with open(os.path.join(HERE, "sheets", "meta.json"), "w", encoding="utf-8") as fp:
    json.dump({"tiles": meta, "frames": [{"seg": s, "i": i, "t": t, "ncand": len(c)} for (im, (s, i, c, t)) in fulls]},
              fp, ensure_ascii=False, indent=1)
print("tiles=%d sheets=%d fulls=%d fullSheets=%d" % (
    len(tiles), (len(tiles) + SHEET_COLS - 1) // SHEET_COLS, len(fulls), (len(fulls) + FULL_COLS - 1) // FULL_COLS))
