# -*- coding: utf-8 -*-
"""统计 RF-DETR-SoccerNet 在锚点帧上的"球"类别误检量(精度侧)。

召回数字容易掩盖代价: RF-DETR 在低阈值下每帧会吐出大量框。本脚本统计:
  - 每帧球类别候选数、其中命中 GT 的(TP, dist<R)、远离 GT 的(FP)
  - 平均每帧球误检数、球候选的精度
用法: python eval_rfdetr_ball_fp.py [--conf 0.05] [--R 0.02]
"""
import json
import math
import os
import sys
import argparse

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rfdetr_soccernet/weights/checkpoint_best_regular.pth")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--R", type=float, default=0.02)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from rf_soccernet_demo import build_model, detect_frame

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.weights, device)

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = 0
    n_ball_dets = 0
    n_tp = 0
    n_fp = 0
    all_dets_per_frame = []
    max_conf_seen = 0.0
    conf_hits = []

    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        if not os.path.exists(dp):
            continue
        d = json.load(open(dp, encoding="utf-8"))
        for a, rec in enumerate(d):
            b = rec.get("ball")
            if not b or not rec.get("real"):
                continue
            fi = int(segs[si][0] * fps) + a * 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            n_frames += 1
            dets = detect_frame(model, fr, device, args.conf)
            ball_dets = [dt for dt in dets if dt[0] == "ball"]
            all_dets_per_frame.append(len(dets))
            tp = fp = 0
            for (_, x1, y1, x2, y2, conf) in ball_dets:
                cx = ((x1 + x2) / 2.0) / W
                cy = ((y1 + y2) / 2.0) / H
                dist = math.hypot(cx - b["x"], cy - b["y"])
                max_conf_seen = max(max_conf_seen, conf)
                if dist < args.R:
                    tp += 1
                    conf_hits.append(conf)
                else:
                    fp += 1
            n_ball_dets += len(ball_dets)
            n_tp += tp
            n_fp += fp
    cap.release()

    avg_all = sum(all_dets_per_frame) / max(1, len(all_dets_per_frame))
    avg_ball = n_ball_dets / max(1, n_frames)
    avg_fp = n_fp / max(1, n_frames)
    prec = 100.0 * n_tp / max(1, n_ball_dets)
    import statistics
    med = statistics.median(conf_hits) if conf_hits else 0.0

    print(f"\n=== RF-DETR 球类别精度统计 (conf>={args.conf}, R={args.R}) ===")
    print(f"锚点帧数:            {n_frames}")
    print(f"平均每帧总框数(4类): {avg_all:.1f}")
    print(f"平均每帧球候选数:    {avg_ball:.2f}")
    print(f"平均每帧球误检(FP):  {avg_fp:.2f}")
    print(f"球候选总数 / TP / FP: {n_ball_dets} / {n_tp} / {n_fp}")
    print(f"球候选精度:          {prec:.1f}%")
    print(f"命中球框置信度中位数: {med:.3f}   观测到的最大球置信度: {max_conf_seen:.3f}")

    out = {
        "conf": args.conf, "R": args.R, "frames": n_frames,
        "avg_dets_per_frame_all_classes": round(avg_all, 1),
        "avg_ball_cands_per_frame": round(avg_ball, 2),
        "avg_ball_fp_per_frame": round(avg_fp, 2),
        "ball_total": n_ball_dets, "ball_tp": n_tp, "ball_fp": n_fp,
        "ball_precision_pct": round(prec, 1),
        "median_hit_conf": round(med, 3),
        "max_ball_conf": round(max_conf_seen, 3),
    }
    os.makedirs(os.path.join(HERE, "rf_out"), exist_ok=True)
    with open(os.path.join(HERE, "rf_out", "rfdetr_ball_fp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("已保存: rf_out/rfdetr_ball_fp.json")


if __name__ == "__main__":
    main()
