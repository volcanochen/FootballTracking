# -*- coding: utf-8 -*-
"""挖掘误检热点 + 统计球的物理速度上限。
1) 三模型融合跑 8 段（锚点帧节奏 a*3），把不匹配真球的检出按位置分格统计
   —— 出现次数多、且位置固定的格 = 静态误检热点（如禁区线/罚球点）
2) 相邻人工锚点（间隔3帧）的位移分布 = 球的实际速度上限，用于跟踪硬约束
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.025
GX, GY = 24, 12


def full_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
    return [(float(b.xywhn[0][0]), float(b.xywhn[0][1]), float(b.conf[0]))
            for b in r.boxes if int(b.cls[0]) == 0]


def tile_predict(model, fr, conf, NX=3, OV=0.20):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    tw = int(UW / (NX - (NX - 1) * OV))
    step = tw - int(tw * OV)
    out = []
    for k in range(NX):
        x0 = min(k * step, UW - tw)
        r = model.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                out.append((float(b.xywh[0][0] + x0) / UW, float(b.xywh[0][1]) / UH,
                            float(b.conf[0])))
    return out


def nms_merge(dets, thr=0.02):
    dets.sort(key=lambda d: -d[2])
    keep = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > thr for k in keep):
            keep.append(d)
    return keep


def main():
    from ultralytics import YOLO
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    tile_w = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, "runs", "ball_tile_v2", "weights", "best.pt")
    m_full = YOLO(os.path.join(HERE, "runs", "ball11n_vb", "weights", "best.pt"))
    m_tile = YOLO(tile_w)
    m_old = YOLO(os.path.join(HERE, "runs", "ball11n", "weights", "best.pt"))
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    grid = {}   # (gx,gy) -> [n_fp, n_tp, conf_sum, segs_set]
    fp_list = []   # (seg, anchor_idx, x, y, conf) 明细供负样本构建
    speeds = []
    n_fp_all = n_tp_all = 0
    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        if not os.path.exists(dp):
            continue
        d = json.load(open(dp, encoding="utf-8"))
        prev = None   # 上一个 real 锚点 (a, x, y)
        for a, rec in enumerate(d):
            fi = int(segs[si][0] * fps) + a * 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                break
            dets = nms_merge(full_predict(m_full, fr, conf) + tile_predict(m_tile, fr, conf)
                             + full_predict(m_old, fr, conf))
            b = rec.get("ball") if rec.get("real") else None
            if b:
                if prev is not None:
                    dt = (a - prev[0]) * 3.0 / fps
                    if dt > 0:
                        speeds.append((math.hypot(b["x"] - prev[1], b["y"] - prev[2]) / dt, a - prev[0]))
                prev = (a, b["x"], b["y"])
            for x, y, sc in dets:
                hit_tp = b and math.hypot(x - b["x"], y - b["y"]) < R
                g = (min(GX - 1, int(x * GX)), min(GY - 1, int(y * GY)))
                e = grid.setdefault(g, [0, 0, 0.0, set()])
                if hit_tp:
                    e[1] += 1
                    n_tp_all += 1
                else:
                    e[0] += 1
                    e[2] += sc
                    e[3].add(si)
                    n_fp_all += 1
                    fp_list.append([si, a, x, y, sc])
        print("seg%d done" % si, flush=True)
    cap.release()

    with open(os.path.join(HERE, "fp_dets.json"), "w", encoding="utf-8") as fp:
        json.dump(fp_list, fp)
    print("\n== FP hotspot cells (top 15, by count) ==")
    for g, e in sorted(grid.items(), key=lambda kv: -kv[1][0])[:15]:
        cx, cy = (g[0] + 0.5) / GX, (g[1] + 0.5) / GY
        print("cell(%d,%d) center=(%.2f,%.2f) FP=%d TP=%d avgConf=%.2f segs=%s"
              % (g[0], g[1], cx, cy, e[0], e[1], e[2] / max(1, e[0]), sorted(e[3])))
    print("total FP=%d TP=%d  FP/TP=%.2f" % (n_fp_all, n_tp_all, n_fp_all / max(1, n_tp_all)))

    if speeds:
        vs = sorted(s for s, _ in speeds)
        print("\n== ball speed (norm units/sec), anchor gaps ==")
        print("n=%d  p50=%.3f p90=%.3f p95=%.3f max=%.3f"
              % (len(vs), vs[len(vs) // 2], vs[int(len(vs) * 0.9)],
                 vs[int(len(vs) * 0.95)], vs[-1]))
        print("gap distribution:", sorted(set(g for _, g in speeds)))


if __name__ == "__main__":
    main()
