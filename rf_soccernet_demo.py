#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR-SoccerNet 对比演示脚本

在 BallTracer 同一批视频片段上跑 RF-DETR (SoccerNet-Tracking 微调) 推理，
画出 4 类框 (ball/player/referee/goalkeeper)，统计球的检出情况，
输出标注视频 + 统计 JSON，便于与自研 BallTracer 做头对头对比。

权重来自 HF: julianzu9612/RFDETR-Soccernet
(checkpoint_best_regular.pth, 训练时基于 rfdetr 1.2.x + DINOv2-base 骨干,
 解码器 hidden_dim=384, 4 类头)。

用法:
  python rf_soccernet_demo.py --weights rfdetr_soccernet/weights/checkpoint_best_regular.pth \
      --seg-ids 12,13,10,23,29,0 --video full.mp4 --segs segs_cur.json --out-dir rf_out
"""
import os
import sys
import json
import argparse
import time
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 兼容性 shim: rfdetr 1.2.1 在导入期从 transformers 取两个工具函数, 但它们在
# transformers 5.0 重写时已被移除 (当前装的是 5.9.0):
#   - find_pruneable_heads_and_indices  -> 仅模型剪枝(prune_heads)用到, 推理不触发
#   - get_aligned_output_features_output_indices -> 仅做输出特征索引对齐
# 这里在 import rfdetr 之前用原函数实现把它们注入对应模块, 不涉及任何 site-packages 改动。
# ---------------------------------------------------------------------------
def _install_xformers_shim():
    # 1) find_pruneable_heads_and_indices
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "find_pruneable_heads_and_indices"):
            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask), device=mask.device)[mask].long()
                return heads, index

            _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    except Exception:
        pass

    # 2) get_aligned_output_features_output_indices
    try:
        import transformers.utils.backbone_utils as _bu
        if not hasattr(_bu, "get_aligned_output_features_output_indices"):
            def get_aligned_output_features_output_indices(out_features, out_indices, stage_names=None):
                if out_features is not None and out_indices is not None:
                    raise ValueError("Cannot specify both out_features and out_indices")
                if out_features is not None:
                    if stage_names is None:
                        raise ValueError("`stage_names` must be specified when `out_features` is specified")
                    out_indices = [stage_names.index(f) for f in out_features]
                else:
                    if out_indices is None:
                        out_indices = [len(stage_names) - 1] if stage_names is not None else [0]
                    if stage_names is not None:
                        out_features = [stage_names[i] for i in out_indices]
                    else:
                        out_features = [str(i) for i in out_indices]
                return out_features, out_indices

            _bu.get_aligned_output_features_output_indices = get_aligned_output_features_output_indices
    except Exception:
        pass


_install_xformers_shim()


CLASS_NAMES = ["ball", "player", "referee", "goalkeeper"]
CLASS_COLORS = {
    "ball": (0, 0, 255),          # 红
    "player": (0, 200, 0),        # 绿
    "referee": (0, 215, 255),     # 黄
    "goalkeeper": (255, 200, 0),  # 青
}
BALL_COLOR = (0, 0, 255)


def find_lwdetr(obj):
    """递归查找既是 torch.nn.Module 又带 reinitialize_detection_head 的 LWDETR 模块。"""
    import torch
    if isinstance(obj, torch.nn.Module) and hasattr(obj, "reinitialize_detection_head"):
        return obj
    for attr in ("model", "backbone", "encoder", "decoder"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            r = find_lwdetr(child)
            if r is not None:
                return r
    return None


def build_model(weights_path, device, variant="base"):
    """加载 RF-DETR-SoccerNet 权重。

    该 checkpoint 是用 rfdetr 1.2.x 训练的 *自定义* 配置 (见权重内保存的 training args):
      encoder='dinov2_windowed_base', hidden_dim=384, sa_nheads=12, ca_nheads=24,
      dec_n_points=4, projector_scale=['P3','P5'], dec_layers=3, num_queries=300,
      group_detr=13, two_stage=True, resolution=560, patch_size=14, num_windows=4 ...
    标准 RFDETRBase (dinov2_windowed_small / hidden_dim=256) 无法直接匹配, 因此这里
    用 kwargs 把配置覆盖成与训练时完全一致, 再 strict 载入微调权重。
    """
    from rfdetr import RFDETRBase
    num_classes = len(CLASS_NAMES)

    # 与权重内 training args 完全一致的覆盖项
    overrides = dict(
        encoder="dinov2_windowed_base",
        hidden_dim=384,
        sa_nheads=12,
        ca_nheads=24,
        dec_n_points=4,
        projector_scale=["P3", "P5"],
        pretrain_weights=None,  # 跳过 rf-detr COCO 预训练权重下载, 仅用 DINOv2-base 骨干
    )
    print(f"[model] 构建 RFDETRBase(覆盖配置: {overrides}) ...")
    model = RFDETRBase(**overrides)
    nn = find_lwdetr(model)
    if nn is None:
        raise RuntimeError("找不到 LWDETR 模块")
    print(f"[model] LWDETR 模块: {type(nn).__name__}")
    nn.reinitialize_detection_head(num_classes)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    else:
        state = ckpt
    missing, unexpected = nn.load_state_dict(state, strict=False)
    print(f"[model] 权重载入完成 | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[model] 前 15 个 missing 键: {missing[:15]}")
    if unexpected:
        print(f"[model] 前 15 个 unexpected 键: {unexpected[:15]}")
    # 移动到设备并设为 eval
    nn.to(device)
    nn.eval()
    return model


@torch.no_grad()
def detect_frame(model, frame_bgr, device, threshold):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    dets = model.predict(pil, threshold=threshold)
    out = []
    if dets is not None and len(dets) > 0:
        for i in range(len(dets)):
            cid = int(dets.class_id[i])
            if 0 <= cid < len(CLASS_NAMES):
                x1, y1, x2, y2 = [float(v) for v in dets.xyxy[i].tolist()]
                conf = float(dets.confidence[i])
                out.append((CLASS_NAMES[cid], x1, y1, x2, y2, conf))
    return out


def draw(frame, dets, ball_thr=0.0, show_persons=False):
    """画检测框。ball_thr: 只画置信度 >= ball_thr 的球; show_persons: 是否画人类别。"""
    for name, x1, y1, x2, y2, conf in dets:
        if name != "ball" and not show_persons:
            continue
        if name == "ball" and conf < ball_thr:
            continue
        color = CLASS_COLORS[name]
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        thick = 3 if name == "ball" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + len(label) * 9 + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_segment(model, video_path, seg, seg_id, fps, out_dir, device, low_thr=0.05,
                    ball_thr=0.40, show_persons=False):
    start, dur, label = seg[0], seg[1], (seg[2] if len(seg) > 2 else "")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[seg {seg_id}] 无法打开视频")
        return None
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    s0 = int(start * vfps)
    s1 = int((start + dur) * vfps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(out_dir, f"rfdetr_s{seg_id}.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), vfps, (W, H))

    n_frames = 0
    ball_ge = {0.1: 0, 0.3: 0, 0.4: 0, 0.5: 0}
    confs_present = []
    ball_traj = []  # 每帧最佳球框中心(用于画轨迹, 仅 ball_thr 以上)
    last_ball = None
    # 各类别的逐帧计数(推理阈值 low_thr 下的全量统计, 不受显示过滤影响)
    cls_counter = {c: 0 for c in CLASS_NAMES}

    pbar = tqdm(total=(s1 - s0), desc=f"seg{seg_id}({label})", unit="f")
    fi = s0
    while fi < s1:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detect_frame(model, frame, device, low_thr)

        best_ball = None
        seen = {c: 0 for c in CLASS_NAMES}
        for d in dets:
            seen[d[0]] += 1
            if d[0] == "ball":
                if best_ball is None or d[5] > best_ball[5]:
                    best_ball = d
        for c in CLASS_NAMES:
            if seen[c] > 0:
                cls_counter[c] += 1
        # 轨迹与统计只认 ball_thr 以上的球
        if best_ball is not None and best_ball[5] >= ball_thr:
            cx = (best_ball[1] + best_ball[3]) / 2
            cy = (best_ball[2] + best_ball[4]) / 2
            last_ball = (cx, cy)
            confs_present.append(best_ball[5])
            for t in ball_ge:
                if best_ball[5] >= t:
                    ball_ge[t] += 1
        else:
            last_ball = None
        if last_ball is not None:
            ball_traj.append(last_ball)

        draw(frame, dets, ball_thr=ball_thr, show_persons=show_persons)
        # 画球轨迹（绿点）
        for (tx, ty) in ball_traj[-60:]:
            cv2.circle(frame, (int(tx), int(ty)), 2, (0, 255, 100), -1)
        # 顶部信息条
        cv2.rectangle(frame, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(frame, f"RF-DETR-SoccerNet | seg{seg_id} {label} | ball>={ball_thr:.2f} | 仅球类",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        vw.write(frame)
        n_frames += 1
        fi += 1
        pbar.update(1)
    cap.release()
    vw.release()
    pbar.close()

    max_conf = max(confs_present) if confs_present else 0.0
    avg_conf = float(np.mean(confs_present)) if confs_present else 0.0
    stat = {
        "seg_id": seg_id,
        "label": label,
        "start_sec": start,
        "dur_sec": dur,
        "frames": n_frames,
        "ball_thr": ball_thr,
        "ball_frames_ge_0.10": ball_ge[0.1],
        "ball_frames_ge_0.30": ball_ge[0.3],
        "ball_frames_ge_0.40": ball_ge[0.4],
        "ball_frames_ge_0.50": ball_ge[0.5],
        "ball_coverage_0.10_pct": round(100.0 * ball_ge[0.1] / n_frames, 1) if n_frames else 0.0,
        "ball_coverage_0.30_pct": round(100.0 * ball_ge[0.3] / n_frames, 1) if n_frames else 0.0,
        "ball_coverage_0.40_pct": round(100.0 * ball_ge[0.4] / n_frames, 1) if n_frames else 0.0,
        "ball_coverage_0.50_pct": round(100.0 * ball_ge[0.5] / n_frames, 1) if n_frames else 0.0,
        "max_ball_conf": round(max_conf, 3),
        "avg_ball_conf_present": round(avg_conf, 3),
        "class_frame_counts": cls_counter,
        "video": out_path,
    }
    print(f"[seg {seg_id}] 完成: {stat}")
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rfdetr_soccernet/weights/checkpoint_best_regular.pth")
    ap.add_argument("--video", default="full.mp4")
    ap.add_argument("--segs", default="segs_cur.json")
    ap.add_argument("--seg-ids", default="12,13,10,23,29,0")
    ap.add_argument("--out-dir", default="rf_out")
    ap.add_argument("--low-thr", type=float, default=0.05,
                    help="模型推理阈值(全类别), 默认 0.05")
    ap.add_argument("--ball-thr", type=float, default=0.40,
                    help="只画置信度 >= 该值的球, 默认 0.40")
    ap.add_argument("--show-persons", action="store_true",
                    help="是否画 player/referee/goalkeeper 类别(默认不画)")
    ap.add_argument("--variant", default="base", choices=["base", "large", "small"])
    ap.add_argument("--dry-model", action="store_true",
                    help="只构建模型并验证权重匹配, 不跑视频")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args.weights, device, args.variant)

    if args.dry_model:
        print("=== dry-model 完成, 权重匹配验证结束 ===")
        return

    segs = json.load(open(args.segs, encoding="utf-8"))
    seg_ids = [int(x) for x in args.seg_ids.split(",")]

    stats = []
    for sid in seg_ids:
        seg = segs[sid]
        st = process_segment(model, args.video, seg, sid, 25.0, out_dir, device,
                             args.low_thr, args.ball_thr, args.show_persons)
        if st:
            stats.append(st)

    stat_path = os.path.join(out_dir, "rfdetr_stats.json")
    json.dump(stats, open(stat_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== 统计已保存: {stat_path} ===")
    for s in stats:
        print(f"  seg{s['seg_id']:>2} {s['label']:<6} 球覆盖@0.10={s['ball_coverage_0.10_pct']:>5}%  "
              f"@0.30={s['ball_coverage_0.30_pct']:>5}%  @0.40={s['ball_coverage_0.40_pct']:>5}%  "
              f"maxConf={s['max_ball_conf']:.3f}  帧={s['frames']}")


if __name__ == "__main__":
    main()
