# -*- coding: utf-8 -*-
"""MPEG 式分区识别：关键帧全画面 + P 帧 ROI 检测，顺序读帧采集候选并缓存。
- I 帧（每 KEY_EVERY 帧一次）：全画面 = 切片模型3分区 + vb 全帧 + old 全帧（5 次推理）
- P 帧：仅预测位置所在分区的切片 + vb 全帧（2 次推理），检出仅作候选（门限阶段过滤）
- 无预测位置时（链未建立）按 I 帧处理
输出 cand_s{si}.json: [{f: 帧号(段内), d: [[x,y,w,h,conf],...]}]
用法: collect_candidates.py <tile_weights> [conf=0.03] [key_every=3]
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
NX, OV = 3, 0.20
KEY_EVERY = 3


def tile_predict(model, up, conf, only_k=None, UH=None, UW=None, tw=0, step=0):
    ks = [only_k] if only_k is not None else range(NX)
    out = []
    for k in ks:
        x0 = min(k * step, UW - tw)
        r = model.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                continue
            bx, by, bw, bh = [float(v) for v in b.xywh[0]]
            out.append(((bx + x0) / UW, by / UH, bw / UW, bh / UH, float(b.conf[0])))
    return out


def full_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
    return [(float(b.xywhn[0][0]), float(b.xywhn[0][1]), float(b.xywhn[0][2]),
             float(b.xywhn[0][3]), float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == 0]


def main():
    from ultralytics import YOLO
    tile_w = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    key_every = int(sys.argv[3]) if len(sys.argv) > 3 else KEY_EVERY
    seg_ids = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2,3,4,5,6,7").split(",")]
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    m_tile = YOLO(tile_w)
    m_full = YOLO(os.path.join(HERE, "runs", "ball11n_vb", "weights", "best.pt"))
    m_old = YOLO(os.path.join(HERE, "runs", "ball11n", "weights", "best.pt"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    for si in seg_ids:
        out_path = os.path.join(HERE, "cand_s%d.json" % si)
        if os.path.exists(out_path):
            print("skip seg%d (cached)" % si, flush=True)
            continue
        start, dur = segs[si][0], segs[si][1]
        n_frames = int(dur * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        # 分区几何（2x 域）
        up0 = cv2.resize(cap.read()[1], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        UH, UW = up0.shape[:2]
        tw = int(UW / (NX - (NX - 1) * OV))
        step = tw - int(tw * OV)
        del up0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

        recs = []
        px = py = None
        n_i = n_p = 0
        for fi in range(n_frames):
            ok, fr = cap.read()
            if not ok:
                break
            # MPEG 式：有预测位置且非关键帧 → P 帧 ROI 模式
            if px is not None and fi % key_every != 0:
                k = min(NX - 1, max(0, int(px * UW // step if step else 0)))
                # 修正边界
                k = max(0, min(NX - 1, int((px * UW) // step)))
                up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                dets = tile_predict(m_tile, up, conf, only_k=k, UH=UH, UW=UW, tw=tw, step=step) \
                    + full_predict(m_full, fr, conf)
                n_p += 1
            else:
                up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                dets = tile_predict(m_tile, up, conf, UH=UH, UW=UW, tw=tw, step=step) \
                    + full_predict(m_full, fr, conf) + full_predict(m_old, fr, conf)
                n_i += 1
            # 中心距 NMS
            dets.sort(key=lambda d: -d[4])
            merged = []
            for d in dets:
                if all(math.hypot(d[0] - k2[0], d[1] - k2[1]) > 0.02 for k2 in merged):
                    merged.append(d)
            recs.append({"f": fi, "d": merged})
            # 维护粗预测位置：取最高分检出（仅供 P 帧选区，精确门限在后级）
            if merged:
                px, py = merged[0][0], merged[0][1]
            del fr, up, dets
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(recs, fp)
        print("seg%d cached: frames=%d I=%d P=%d" % (si, len(recs), n_i, n_p), flush=True)
    cap.release()
    print("collect done")


if __name__ == "__main__":
    main()
