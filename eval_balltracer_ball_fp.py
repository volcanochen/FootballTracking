# -*- coding: utf-8 -*-
"""BallTracer 同口径球类别精度统计 (与 eval_rfdetr_ball_fp.py 对齐)。

协议完全一致:
  - 同一批 181 个锚点帧 (dets_track, ball 非空且 real=True)
  - 单帧推理, conf=0.05
  - 球候选: 3 模型融合 (ball11n_vb + ball_tile_v3 + ball11n) 经 NMS(thr=0.02)
  - 命中判定: 归一化中心距离 < R=0.02
用法: python eval_balltracer_ball_fp.py [--conf 0.05] [--R 0.02]
"""
import json
import math
import os
import sys
import argparse

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def full_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
    return [(float(b.xywhn[0][0]), float(b.xywhn[0][1]), float(b.conf[0]))
            for b in r.boxes if int(b.cls[0]) == 0]


def tile_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    NX, OV = 3, 0.20
    tw = int(UW / (NX - (NX - 1) * OV)); step = tw - int(tw * OV)
    out = []
    for k in range(NX):
        x0 = min(k * step, UW - tw)
        r = model.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                out.append((float(b.xywh[0][0] + x0) / UW, float(b.xywh[0][1]) / UH, float(b.conf[0])))
    return out


def nms_merge(dets, thr=0.02):
    dets.sort(key=lambda d: -d[2])
    keep = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > thr for k in keep):
            keep.append(d)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--R", type=float, default=0.02)
    args = ap.parse_args()

    from ultralytics import YOLO
    m_full = YOLO(os.path.join(HERE, 'runs', 'ball11n_vb', 'weights', 'best.pt'))
    m_tile = YOLO(os.path.join(HERE, 'runs', 'ball_tile_v3', 'weights', 'best.pt'))
    m_old = YOLO(os.path.join(HERE, 'runs', 'ball11n', 'weights', 'best.pt'))

    segs = json.load(open(os.path.join(HERE, 'segs_cur.json'), encoding='utf-8'))
    cap = cv2.VideoCapture(os.path.join(HERE, 'full.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    n_frames = 0
    n_ball_dets = 0
    n_tp = 0
    n_fp = 0
    max_conf_seen = 0.0
    conf_hits = []
    per = {}
    # 阈值曲线: 记录每锚点最佳命中框 conf
    best_hit_conf = []

    for si in range(8):
        dp = os.path.join(HERE, 'dets_track', '%d.json' % si)
        if not os.path.exists(dp):
            continue
        d = json.load(open(dp, encoding='utf-8'))
        st = [0, 0]
        for a, rec in enumerate(d):
            b = rec.get('ball')
            if not b or not rec.get('real'):
                continue
            fi = int(segs[si][0] * fps) + a * 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            n_frames += 1
            dets = nms_merge(full_predict(m_full, fr, args.conf)
                             + tile_predict(m_tile, fr, args.conf)
                             + full_predict(m_old, fr, args.conf))
            tp = fp = 0
            best = 0.0
            for (x, y, sc) in dets:
                dist = math.hypot(x - b['x'], y - b['y'])
                max_conf_seen = max(max_conf_seen, sc)
                if dist < args.R:
                    tp += 1
                    best = max(best, sc)
                    conf_hits.append(sc)
                else:
                    fp += 1
            n_ball_dets += len(dets)
            n_tp += tp
            n_fp += fp
            best_hit_conf.append(best)
            st[1] += 1
            if tp > 0:
                st[0] += 1
            del fr
        per[si] = st
    cap.release()

    avg_ball = n_ball_dets / max(1, n_frames)
    avg_fp = n_fp / max(1, n_frames)
    prec = 100.0 * n_tp / max(1, n_ball_dets)
    import statistics
    med = statistics.median(conf_hits) if conf_hits else 0.0

    print(f"\n=== BallTracer 球类别精度统计 (conf>={args.conf}, R={args.R}, 单帧) ===")
    for k in sorted(per):
        print('seg%d: %d/%d (%.0f%%)' % (k, per[k][0], per[k][1], 100.0 * per[k][0] / max(1, per[k][1])))
    print(f"锚点帧数:            {n_frames}")
    print(f"平均每帧球候选数:    {avg_ball:.2f}")
    print(f"平均每帧球误检(FP):  {avg_fp:.2f}")
    print(f"球候选总数 / TP / FP: {n_ball_dets} / {n_tp} / {n_fp}")
    print(f"球候选精度:          {prec:.1f}%")
    print(f"命中球框置信度中位数: {med:.3f}   观测到的最大球置信度: {max_conf_seen:.3f}")
    print(f"单帧命中锚点数:      {sum(1 for c in best_hit_conf if c > 0)}/{n_frames}")

    # 阈值曲线 (与 RF-DETR 对齐)
    curve = {}
    for thr in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        curve[str(thr)] = round(100.0 * sum(1 for c in best_hit_conf if c >= thr) / max(1, n_frames), 1)
    print(f"recall_by_threshold: {curve}")
    print(f"recall_no_threshold: {100.0 * sum(1 for c in best_hit_conf if c > 0) / max(1, n_frames):.1f}")

    out = {
        "conf": args.conf, "R": args.R, "frames": n_frames,
        "avg_ball_cands_per_frame": round(avg_ball, 2),
        "avg_ball_fp_per_frame": round(avg_fp, 2),
        "ball_total": n_ball_dets, "ball_tp": n_tp, "ball_fp": n_fp,
        "ball_precision_pct": round(prec, 1),
        "median_hit_conf": round(med, 3),
        "max_ball_conf": round(max_conf_seen, 3),
        "recall_by_threshold": curve,
        "recall_no_threshold": round(100.0 * sum(1 for c in best_hit_conf if c > 0) / max(1, n_frames), 1),
    }
    os.makedirs(os.path.join(HERE, 'rf_out'), exist_ok=True)
    with open(os.path.join(HERE, 'rf_out', 'balltracer_ball_fp.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("已保存: rf_out/balltracer_ball_fp.json")


if __name__ == '__main__':
    main()
