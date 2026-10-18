# -*- coding: utf-8 -*-
"""候选采集：MPEG 式分区识别。

- I 帧（每 KEY_EVERY 帧一次 / 链未建立时）：全画面 = 切片模型 3 分区 + 全帧 vb + 全帧 old（5 次推理）
- P 帧：仅预测位置所在分区的切片 + 全帧 vb（2 次推理），检出仅作候选（门限阶段过滤）
- 顺序读帧，避免 cv2 反复 seek 的帧错位
输出 cand_s{seg_id}.json: [{f: 段内帧号, d: [[x,y,w,h,conf], ...]}]
"""
import json
import math
import os

import cv2


def tile_predict(model, up, conf, only_k=None, UH=None, UW=None, tw=0, step=0):
    """对 2x 上采样画面的某一列切片做检测，坐标归一化到 (UW, UH)。"""
    ks = [only_k] if only_k is not None else range(3)
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
    """全帧（2x 上采样）检测，坐标归一化到 2x 画面。"""
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
    return [(float(b.xywhn[0][0]), float(b.xywhn[0][1]), float(b.xywhn[0][2]),
             float(b.xywhn[0][3]), float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == 0]


def collect_segment(m_tile, m_full, m_old, video, seg, seg_id, conf, key_every, out_dir):
    from .config import NX, OV
    start, dur = seg[0], seg[1]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n_frames = int(dur * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
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
        # MPEG 式：有预测位置且非关键帧 -> P 帧 ROI 模式
        if px is not None and fi % key_every != 0:
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
        if merged:
            px, py = merged[0][0], merged[0][1]
        del fr, up, dets
    cap.release()

    out_path = os.path.join(out_dir, "cand_s%d.json" % seg_id)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(recs, fp)
    return n_frames, n_i, n_p


def collect_candidates(video, segs, seg_ids, tile_weights, full_weights=None,
                      old_weights=None, conf=0.03, key_every=3, out_dir=".", skip_cached=True):
    """采集多个段的候选并缓存到 out_dir/cand_s{id}.json。

    返回 [(seg_id, n_frames | "cached"), ...]
    """
    from ultralytics import YOLO
    from .config import DEFAULT_FULL_WEIGHTS, DEFAULT_OLD_WEIGHTS
    m_tile = YOLO(tile_weights)
    m_full = YOLO(full_weights or DEFAULT_FULL_WEIGHTS)
    m_old = YOLO(old_weights or DEFAULT_OLD_WEIGHTS)

    results = []
    for si in seg_ids:
        out_path = os.path.join(out_dir, "cand_s%d.json" % si)
        if skip_cached and os.path.exists(out_path):
            print("skip seg%d (cached)" % si, flush=True)
            results.append((si, "cached"))
            continue
        nf, ni, np_ = collect_segment(m_tile, m_full, m_old, video, segs[si], si, conf, key_every, out_dir)
        print("seg%d cached: frames=%d I=%d P=%d" % (si, nf, ni, np_), flush=True)
        results.append((si, nf))
    print("collect done")
    return results
