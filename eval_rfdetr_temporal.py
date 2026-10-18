# -*- coding: utf-8 -*-
"""RF-DETR-SoccerNet 在人工锚点上的"时序融合版"召回评测。

与 eval_anchor_temporal.py (BallTracer 85.1% 的口径) 完全对齐:
  - R = 0.03
  - 邻帧 radius=1 (t-1, t, t+1), 邻帧置信度衰减 0.6
  - nms_merge 去重 (阈值 0.02)
  - conf = 0.05
唯一差异: BallTracer 是 3 个 YOLO 模型融合(full + tile + old)再叠时序,
这里是单个 RF-DETR 模型叠完全相同的时序融合, 属于系统级公平对比。

用法: python eval_rfdetr_temporal.py [conf=0.05] [radius=1]
"""
import json
import math
import os
import sys
import argparse

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.03
NMS_THR = 0.02


def nms_merge(dets, thr=NMS_THR):
    dets = sorted(dets, key=lambda d: -d[2])
    keep = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > thr for k in keep):
            keep.append(d)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rfdetr_soccernet/weights/checkpoint_best_regular.pth")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--radius", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from rf_soccernet_demo import build_model, detect_frame

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = build_model(args.weights, device)

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tot = hit = 0
    per = {}
    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        if not os.path.exists(dp):
            continue
        d = json.load(open(dp, encoding="utf-8"))
        st = [0, 0]
        for a, rec in enumerate(d):
            b = rec.get("ball")
            if not b or not rec.get("real"):
                continue
            base = int(segs[si][0] * fps) + a * 3
            all_dets = []
            for off in range(-args.radius, args.radius + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, base + off)
                ok, fr = cap.read()
                if not ok:
                    continue
                w_decay = 1.0 if off == 0 else 0.6
                for (name, x1, y1, x2, y2, conf) in detect_frame(model, fr, device, args.conf):
                    if name != "ball":
                        continue
                    cx = ((x1 + x2) / 2.0) / W
                    cy = ((y1 + y2) / 2.0) / H
                    all_dets.append((cx, cy, conf * w_decay))
                del fr
            dets = nms_merge(all_dets)
            st[1] += 1
            best = min((math.hypot(x - b["x"], y - b["y"]) for x, y, sc in dets), default=9)
            if best < R:
                st[0] += 1
        per[si] = st
        tot += st[1]
        hit += st[0]
        print(f"  seg{si}: {st[0]}/{st[1]}")
    cap.release()

    for k in sorted(per):
        print("seg%d: %d/%d (%.0f%%)" % (k, per[k][0], per[k][1],
                                         100.0 * per[k][0] / max(1, per[k][1])))
    print("RF-DETR TEMPORAL(r=%d) conf=%.2f recall: %d/%d (%.1f%%)"
          % (args.radius, args.conf, hit, tot, 100.0 * hit / max(1, tot)))

    out = {
        "protocol": {"R": R, "radius": args.radius, "conf": args.conf,
                     "nms_thr": NMS_THR, "neighbor_decay": 0.6},
        "total_anchors": tot,
        "hit": hit,
        "recall_pct": round(100.0 * hit / max(1, tot), 1),
    }
    with open(os.path.join(HERE, "rf_out", "rfdetr_temporal_recall.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("已保存: rf_out/rfdetr_temporal_recall.json")


if __name__ == "__main__":
    main()
