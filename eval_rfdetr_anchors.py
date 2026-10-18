# -*- coding: utf-8 -*-
"""RF-DETR-SoccerNet 在人工锚点上的召回评测（与 BallTracer 评测口径完全一致）。

BallTracer 的 85.1% 召回是用 eval_anchor_recall.py 算的, 口径如下:
  - 分母: dets_track/{0..7}.json 中 ball 非空 且 real=True 的锚点(球确定可见的帧)
  - 帧号: fi = int(segs[si][0] * fps) + a * 3   (a 为锚点序号)
  - 命中: 检出的球中心与 GT 中心的归一化距离 < R = 0.02

本脚本用完全相同的分母 / 帧号 / 命中判据评测 RF-DETR-SoccerNet, 从而得到
可直接对比的数字。推理只跑一遍(低阈值取回全部候选), 再按不同置信度阈值
分别统计召回, 得到阈值-召回曲线。

用法: python eval_rfdetr_anchors.py [--weights ...] [--min-conf 0.01]
"""
import os
import sys
import json
import math
import argparse

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.02  # 与 eval_anchor_recall.py 一致
CLASS_BALL = "ball"

# 评测用的置信度阈值档位 (0.10 是 BallTracer 评测时用的 conf, 0.40 是该仓库
# config.json 推荐的 ball 阈值)
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rfdetr_soccernet/weights/checkpoint_best_regular.pth")
    ap.add_argument("--min-conf", type=float, default=0.01,
                    help="推理时的最低阈值, 取回全部候选后再按各档位统计")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # 复用 demo 脚本的建模型函数(内含 transformers 兼容 shim 与精确架构覆盖)
    sys.path.insert(0, HERE)
    from rf_soccernet_demo import build_model, detect_frame

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = build_model(args.weights, device)

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video {W}x{H} fps={fps:.2f}")

    # 收集: 每个锚点 -> 所有球候选的 (归一化距离, 置信度)
    anchors = []  # [(seg, anchor_idx, gt_x, gt_y)]
    results = []  # [[(dist, conf), ...], ...] 与 anchors 一一对应

    total_anchors = 0
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
            dets = detect_frame(model, fr, device, args.min_conf)
            cands = []
            for (name, x1, y1, x2, y2, conf) in dets:
                if name != CLASS_BALL:
                    continue
                cx = ((x1 + x2) / 2.0) / W
                cy = ((y1 + y2) / 2.0) / H
                cands.append((math.hypot(cx - b["x"], cy - b["y"]), conf))
            anchors.append((si, a, b["x"], b["y"]))
            results.append(cands)
            total_anchors += 1
        print(f"  seg{si} 锚点已评测 (累计 {total_anchors})")
    cap.release()

    print(f"\n=== 可见锚点总数: {total_anchors} (与 BallTracer 评测分母一致) ===")

    # 各阈值档位的召回
    print("\n阈值   命中   召回")
    for T in THRESHOLDS:
        hit = 0
        for cands in results:
            for dist, conf in cands:
                if conf >= T and dist < R:
                    hit += 1
                    break
        print(f"{T:>5.2f}  {hit:>4}/{total_anchors}  {100.0*hit/max(1,total_anchors):>5.1f}%")

    # 无置信度限制(只看是否存在几何上足够近的球框)
    hit_any = sum(1 for cands in results if any(d < R for d, _ in cands))
    print(f"{'无限制':>5}  {hit_any:>4}/{total_anchors}  {100.0*hit_any/max(1,total_anchors):>5.1f}%")

    # 每分段明细(取 0.10 与 0.40 两档)
    print("\n=== 分段明细 ===")
    for T in (0.10, 0.40):
        print(f"--- 阈值 {T} ---")
        agg = {}
        for (si, a, gx, gy), cands in zip(anchors, results):
            h = any(dist < R and conf >= T for dist, conf in cands)
            s = agg.setdefault(si, [0, 0])
            s[1] += 1
            s[0] += int(h)
        for k in sorted(agg):
            print(f"  seg{k}: {agg[k][0]}/{agg[k][1]} ({100.0*agg[k][0]/max(1,agg[k][1]):.0f}%)")

    # 保存明细便于后续写报告
    out = {
        "total_anchors": total_anchors,
        "R": R,
        "recall_by_threshold": {
            str(T): round(100.0 * sum(1 for c in results if any(d < R and cf >= T for d, cf in c))
                          / max(1, total_anchors), 1)
            for T in THRESHOLDS
        },
        "recall_no_threshold": round(100.0 * hit_any / max(1, total_anchors), 1),
    }
    with open(os.path.join(HERE, "rf_out", "rfdetr_anchor_recall.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: rf_out/rfdetr_anchor_recall.json")


if __name__ == "__main__":
    main()
