# -*- coding: utf-8 -*-
"""切片域模型 + 双向时序门限跟踪 的标注视频（tile 版）。
检测：2x 上采样后左右 3 切片(20%重叠)各 1280 推理，NMS 合并 → 有效分辨率 ~4.5x。
跟踪/门限/图例与 anno_track_video.py 相同：
红框=门限内检出(标conf) / 橙框pred=插值 / 灰虚线=双向均拒突现检出 / 绿点=轨迹尾巴
用法: python anno_track_video_tile.py <weights> [seg_ids=2,3] [conf=0.10]
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_BASE = 0.02
GATE_V = 1.5
GATE_MAX = 0.04       # 物理速度硬约束：球速 p95=0.49 宽/s ≈ 0.016/帧，0.04/帧已留 2.5x 余量
RECAP_STEP = 0.003
RECAP_MAX = 0.04
MAXLOST = 25
INIT_CONF = 0.30
FILL_MAX = 15
JUMP_HARD = 0.06      # 链路级清理：相邻两点位移超过此值（=1.8宽/s）视为不可能跳变
NX, OV = 3, 0.20


def full_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
    return [(float(b.xywhn[0][0]), float(b.xywhn[0][1]), float(b.xywhn[0][2]),
             float(b.xywhn[0][3]), float(b.conf[0]))
            for b in r.boxes if int(b.cls[0]) == 0]


def tile_predict(model, fr, conf):
    H, W = fr.shape[:2]
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
                continue
            bx, by, bw, bh = [float(v) for v in b.xywh[0]]
            out.append(((bx + x0) / UW, by / UH, bw / UW, bh / UH, float(b.conf[0])))
    out.sort(key=lambda d: -d[4])
    keep = []
    for c in out:
        okk = True
        for k in keep:
            ix = max(0, min(c[0] + c[2] / 2, k[0] + k[2] / 2) - max(c[0] - c[2] / 2, k[0] - k[2] / 2))
            iy = max(0, min(c[1] + c[3] / 2, k[1] + k[3] / 2) - max(c[1] - c[3] / 2, k[1] - k[3] / 2))
            uni = c[2] * c[3] + k[2] * k[3] - ix * iy
            if uni > 0 and ix * iy / uni > 0.45:
                okk = False
                break
        if okk:
            keep.append(c)
    return keep


def run_pass(cands, n, reverse=False):
    track, rejected = {}, {}
    px = py = None
    vx = vy = 0.0
    lost = 0
    order = range(n - 1, -1, -1) if reverse else range(n)
    for fi in order:
        cs = cands[fi] if fi < len(cands) else []
        if px is None:
            strong = [c for c in cs if c[4] >= INIT_CONF]
            if strong:
                track[fi] = strong[0]
                px, py, vx, vy, lost = strong[0][0], strong[0][1], 0.0, 0.0, 0
            continue
        ex, ey = px + vx, py + vy
        R = min(GATE_MAX, GATE_BASE + GATE_V * math.hypot(vx, vy) + RECAP_STEP * lost)
        best, best_d = None, 1e9
        for c in cs:
            d = math.hypot(c[0] - ex, c[1] - ey)
            if d <= R and d < best_d:
                best, best_d = c, d
        if best:
            track[fi] = best
            vx = 0.65 * vx + 0.35 * (best[0] - px)
            vy = 0.65 * vy + 0.35 * (best[1] - py)
            px, py, lost = best[0], best[1], 0
        else:
            vx, vy = vx * 0.8, vy * 0.8
            ex2, ey2 = px + vx, py + vy
            R2 = min(GATE_MAX, GATE_BASE + GATE_V * math.hypot(vx, vy) + RECAP_STEP * (lost + 1))
            rej = [c for c in cs if math.hypot(c[0] - ex2, c[1] - ey2) > R2][:3]
            if rej:
                rejected[fi] = rej
            lost += 1
            if lost > MAXLOST:
                px = py = None
    return track, rejected


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    weights = sys.argv[1]
    seg_ids = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,3").split(",")]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    model = YOLO(weights)
    # 融合辅助模型：全帧 vb + 旧 ft 模型（低分高召回），NMS 合并
    m_full = YOLO(os.path.join(HERE, "runs", "ball11n_vb", "weights", "best.pt"))
    m_old = YOLO(os.path.join(HERE, "runs", "ball11n", "weights", "best.pt"))
    tag = os.path.basename(os.path.dirname(os.path.dirname(weights))) + "_ens"
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    for si in seg_ids:
        start, dur = segs[si][0], segs[si][1]
        n_frames = int(dur * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        cands = []
        for _ in range(n_frames):
            ok, fr = cap.read()
            if not ok:
                break
            dets = tile_predict(model, fr, conf) + full_predict(m_full, fr, conf) \
                + full_predict(m_old, fr, conf)
            # 中心距 NMS 合并（归一化坐标）
            dets.sort(key=lambda d: -d[4])
            merged = []
            for d in dets:
                if all(math.hypot(d[0] - k[0], d[1] - k[1]) > 0.02 for k in merged):
                    merged.append(d)
            cands.append(merged)
            del fr

        tf, rf_ = run_pass(cands, n_frames, False)
        tb, rb_ = run_pass(cands, n_frames, True)
        track = dict(tf)
        for k, v in tb.items():
            if k not in track:
                track[k] = v
        # 长时轴清理：前后向并集里可能出现缝合跳变，按物理速度上限剔除
        while True:
            ks = sorted(track)
            bad = None
            for a, b in zip(ks, ks[1:]):
                if math.hypot(track[b][0] - track[a][0], track[b][1] - track[a][1]) > JUMP_HARD * (b - a):
                    # 剔除置信度较低的一端
                    bad = b if track[b][4] < track[a][4] else a
                    break
            if bad is None:
                break
            del track[bad]
        rejected = {}
        for k in set(rf_) | set(rb_):
            if k not in track and k in rf_ and k in rb_:
                common = [c for c in rf_[k] if c in rb_[k]]
                if common:
                    rejected[k] = common
        ks = sorted(track)
        fills = {}
        for a, b in zip(ks, ks[1:]):
            gap = b - a - 1
            if 0 < gap <= FILL_MAX:
                ta, tb2 = track[a], track[b]
                for j in range(a + 1, b):
                    u = (j - a) / (b - a)
                    e = u * u * (3 - 2 * u)
                    fills[j] = (ta[0] + (tb2[0] - ta[0]) * e, ta[1] + (tb2[1] - ta[1]) * e)

        out_path = os.path.join(HERE, "model_track_s%d_%s.mp4" % (si, tag))
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (854, 356))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        trail = []
        H, W = 356, 854
        for fi in range(n_frames):
            ok, fr = cap.read()
            if not ok:
                break

            def draw_box(x, y, w, h, color, thick, dashed=False):
                x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
                x2, y2 = x1 + max(5, int(w * W)), y1 + max(5, int(h * H))
                if not dashed:
                    cv2.rectangle(fr, (x1, y1), (x2, y2), color, thick)
                    return
                for a, b2 in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                              ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
                    L = max(abs(b2[0] - a[0]), abs(b2[1] - a[1]))
                    for t in range(0, L, 6):
                        if a[1] == b2[1]:
                            p1 = (a[0] + t, a[1]); p2 = (min(a[0] + t + 3, b2[0]), a[1])
                        else:
                            p1 = (a[0], a[1] + t); p2 = (a[0], min(a[1] + t + 3, b2[1]))
                        cv2.line(fr, p1, p2, color, 1)

            drawn = False
            if fi in track:
                x, y, w, h, sc = track[fi]
                draw_box(x, y, w, h, (0, 0, 255), 2)
                cv2.putText(fr, "ball %.2f" % sc, (int(x * W) + 6, int(y * H) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                trail.append((int(x * W), int(y * H)))
                drawn = True
            elif fi in fills:
                x, y = fills[fi]
                draw_box(x, y, 0.012, 0.02, (0, 150, 255), 1)
                cv2.putText(fr, "pred", (int(x * W) + 6, int(y * H) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
                trail.append((int(x * W), int(y * H)))
                drawn = True
            if not drawn:
                for c in rejected.get(fi, []):
                    draw_box(c[0], c[1], c[2], c[3], (160, 160, 160), 1, dashed=True)
            if len(trail) > 18:
                trail.pop(0)
            for ti in range(1, len(trail)):
                cv2.circle(fr, trail[ti], 1, (0, 200, 120), -1)
            cv2.putText(fr, "seg%d %.1fs" % (si, fi / fps), (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            wr.write(fr)
        wr.release()
        n_rej = sum(len(v) for v in rejected.values())
        print("saved %s" % out_path, flush=True)
        print("  frames=%d det=%d(%.0f%%) interp=%d rejected_far=%d empty=%d" % (
            n_frames, len(track), 100.0 * len(track) / n_frames, len(fills),
            n_rej, n_frames - len(track) - len(fills)), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
