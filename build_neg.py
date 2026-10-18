# -*- coding: utf-8 -*-
"""根据 mine_fp.py 导出的 fp_dets.json 构建定向难例负样本：
- 误检帧（含高置信误检的位置）切片后加入训练集
- 若该帧同时有人工确认真球，则含球切片带标签、其余切片为空标签（背景）
- 无真球帧 → 3 个切片全为空标签（纯背景，重点压制禁区线等静态误检）
"""
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
NX, OV = 3, 0.20
UP = 2.0
MIN_CONF = 0.10
MAX_FRAMES = 600


def tiles_of(w2):
    tw = int(w2 / (NX - (NX - 1) * OV))
    step = tw - int(tw * OV)
    return [(min(k * step, w2 - tw), tw) for k in range(NX)]


def emit(img2, boxes, name, img_d, lbl_d):
    H2, W2 = img2.shape[:2]
    for k, (x0, tw) in enumerate(tiles_of(W2)):
        tile = img2[:, x0:x0 + tw]
        th = tile.shape[0]
        lines = []
        for (bx, by, bw, bh) in boxes:
            cx, cy = bx * W2, by * H2
            if not (x0 < cx < x0 + tw):
                continue
            w_abs, h_abs = bw * W2, bh * H2
            if cx - w_abs / 2 < x0 - 2 or cx + w_abs / 2 > x0 + tw + 2:
                continue
            lines.append("0 %.6f %.6f %.6f %.6f" % ((cx - x0) / tw, cy / th,
                                                    bw * W2 / tw, bh * H2 / th))
        cv2.imwrite(os.path.join(img_d, "%s_t%d.jpg" % (name, k)), tile)
        with open(os.path.join(lbl_d, "%s_t%d.txt" % (name, k)), "w") as fp:
            fp.write("\n".join(lines))


def main():
    fp = json.load(open(os.path.join(HERE, "fp_dets.json"), encoding="utf-8"))
    # 聚合到帧级：seg -> anchor -> [最高conf, 各检出]
    by_frame = {}
    for si, a, x, y, sc in fp:
        by_frame.setdefault((si, a), []).append((x, y, sc))

    # 优先高置信误检帧
    frames = sorted(by_frame, key=lambda k: -max(c for _, _, c in by_frame[k]))
    frames = [f for f in frames if max(c for _, _, c in by_frame[f]) >= MIN_CONF][:MAX_FRAMES]
    print("negative frames to emit: %d" % len(frames))

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    anchors = {}
    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        if os.path.exists(dp):
            anchors[si] = json.load(open(dp, encoding="utf-8"))

    val_frames = set()
    vdir = os.path.join(HERE, "dataset2", "val", "images")
    for fn in os.listdir(vdir):
        p = fn[:-4].split("_")
        val_frames.add((int(p[0][1:]), int(p[1][1:]) * 3))

    img_d = os.path.join(HERE, "dataset3", "train", "images")
    lbl_d = os.path.join(HERE, "dataset3", "train", "labels")
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n = 0
    for (si, a) in frames:
        if (si, a * 3) in val_frames:
            continue
        fi = int(segs[si][0] * fps) + a * 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        boxes = []
        rec = anchors.get(si, [None] * 400)[a] if a < len(anchors.get(si, [])) else None
        if rec and rec.get("real") and rec.get("ball"):
            b = rec["ball"]
            boxes = [(b["x"], b["y"], 0.015, 0.015)]
        up = cv2.resize(fr, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
        emit(up, boxes, "ng_s%d_i%03d" % (si, a), img_d, lbl_d)
        n += 1
    cap.release()
    print("emitted %d negative frames x3 tiles" % n)
    for d in (img_d, lbl_d):
        os.remove(os.path.join(HERE, "dataset3", "train", "labels.cache")) if False else None
    for cache in ("dataset3/train/labels.cache", "dataset3/val/labels.cache"):
        p = os.path.join(HERE, cache)
        if os.path.exists(p):
            os.remove(p)
            print("removed", cache)


if __name__ == "__main__":
    main()
