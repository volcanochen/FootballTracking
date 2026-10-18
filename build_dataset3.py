# -*- coding: utf-8 -*-
"""构建切片训练集 dataset3：
1) dataset2 的 141 train / 29 val 图 → 2x 上采样 → 左中右 3 切片（20% 重叠）
2) 伪标签：当前 vb 模型对 8 段逐帧推理，conf>=0.5 的检出加入（扩充数据量）
   - val 帧所在的 (seg, 帧) 跳过，防止验证集泄漏
输出 dataset3/{train,val}/{images,labels}
"""
import json
import math
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
NX, OV = 3, 0.20
PSEUDO_CONF = 0.5
UP = 2.0


def tiles_of(w2):
    tw = int(w2 / (NX - (NX - 1) * OV))
    step = tw - int(tw * OV)
    return [(min(k * step, w2 - tw), tw) for k in range(NX)]


def emit(img2, boxes, split, name, out_dirs):
    """boxes: 2x图归一化 [(x,y,w,h)] -> 每个切片一份 YOLO 标签"""
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
                continue   # 框被切边，丢弃
            nx = (cx - x0) / tw
            ny = cy / th
            lines.append("0 %.6f %.6f %.6f %.6f" % (nx, ny, bw * W2 / tw, bh * H2 / th))
        cv2.imwrite(os.path.join(out_dirs[split]["img"], "%s_t%d.jpg" % (name, k)), tile)
        with open(os.path.join(out_dirs[split]["lbl"], "%s_t%d.txt" % (name, k)), "w") as fp:
            fp.write("\n".join(lines))


def main():
    from ultralytics import YOLO
    d3 = os.path.join(HERE, "dataset3")
    out = {}
    for split in ("train", "val"):
        img_d = os.path.join(d3, split, "images")
        lbl_d = os.path.join(d3, split, "labels")
        os.makedirs(img_d, exist_ok=True)
        os.makedirs(lbl_d, exist_ok=True)
        out[split] = {"img": img_d, "lbl": lbl_d}

    # ---- 1) dataset2 -> tiles ----
    for split in ("train", "val"):
        src_img = os.path.join(HERE, "dataset2", split, "images")
        src_lbl = os.path.join(HERE, "dataset2", split, "labels")
        n_t = 0
        for fn in sorted(os.listdir(src_img)):
            stem = os.path.splitext(fn)[0]
            img = cv2.imread(os.path.join(src_img, fn))
            H, W = img.shape[:2]
            img2 = cv2.resize(img, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
            boxes = []
            lf = os.path.join(src_lbl, stem + ".txt")
            if os.path.exists(lf):
                for line in open(lf):
                    p = line.split()
                    if len(p) == 5:
                        boxes.append(tuple(float(v) for v in p[1:]))
            emit(img2, boxes, split, stem, out)
            n_t += 1
        print("%s: %d source imgs tiled" % (split, n_t), flush=True)

    # ---- 2) 难例挖掘：原管线人工确认锚点（real=True）全部加入 ----
    val_names = set(os.listdir(os.path.join(HERE, "dataset2", "val", "images")))
    val_frames = set()   # (seg, frame_off) 防泄漏
    for fn in val_names:
        p = fn[:-4].split("_")   # s4_i0006
        seg, i = int(p[0][1:]), int(p[1][1:])
        val_frames.add((seg, i * 3))
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n_anch = 0
    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        if not os.path.exists(dp):
            continue
        d = json.load(open(dp, encoding="utf-8"))
        for a, rec in enumerate(d):
            b = rec.get("ball")
            if not b or not rec.get("real"):
                continue
            tgt = int(segs[si][0] * fps) + a * 3
            if (si, a * 3) in val_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
            ok, fr = cap.read()
            if not ok:
                continue
            up = cv2.resize(fr, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
            emit(up, [(b["x"], b["y"], 0.015, 0.015)], "train", "an_s%d_i%03d" % (si, a), out)
            n_anch += 1
        print("seg%d anchors added" % si, flush=True)
    print("anchor frames added: %d" % n_anch, flush=True)

    # ---- 3) 伪标签（仅 train）----
    model = YOLO(os.path.join(HERE, "runs", "ball11n_vb", "weights", "best.pt"))
    n_pos = 0
    for si in range(len(segs)):
        start, dur = segs[si][0], segs[si][1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        for fi in range(int(dur * fps)):
            if (si, fi) in val_frames:
                cap.read()
                continue
            ok, fr = cap.read()
            if not ok:
                break
            up = cv2.resize(fr, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
            r = model.predict(up, imgsz=1280, conf=PSEUDO_CONF, verbose=False)[0]
            boxes = []
            for b in r.boxes:
                if int(b.cls[0]) != 0:
                    continue
                boxes.append(tuple(float(v) for v in b.xywhn[0]))
            if boxes:
                emit(up, boxes, "train", "ps_s%d_f%04d" % (si, fi), out)
                n_pos += 1
            del up, r
        print("seg%d pseudo done, pos_frames=%d" % (si, n_pos), flush=True)
    cap.release()

    with open(os.path.join(d3, "ball.yaml"), "w", encoding="utf-8") as fp:
        fp.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n" % d3.replace("\\", "/"))
    n_tr = len(os.listdir(out["train"]["img"]))
    n_va = len(os.listdir(out["val"]["img"]))
    print("dataset3: train tiles=%d val tiles=%d pseudo_pos_frames=%d" % (n_tr, n_va, n_pos))


if __name__ == "__main__":
    main()
