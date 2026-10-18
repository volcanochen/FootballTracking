# -*- coding: utf-8 -*-
"""最终标注视频：候选缓存 + 双向时序门限(物理速度约束) + 长时轴跳变清理 + 插值。
图例: 红框=门限内检出(标conf) / 橙框pred=插值 / 灰虚线=双向均拒的突现检出 / 绿点=轨迹尾巴
用法: anno_track_video_v2.py [seg_ids=0,1,2,3,4,5,6,7] [conf_min=0.05]
"""
import json
import math
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_BASE = 0.02
GATE_V = 1.5
GATE_MAX = 0.04       # 物理速度硬约束：球速 p95≈0.016/帧，0.04/帧已留 2.5x 余量
RECAP_STEP = 0.003
MAXLOST = 25
INIT_CONF = 0.30
FILL_MAX = 15
JUMP_HARD = 0.06      # 链路级清理：相邻位移超过 1.8宽/s 视为不可能跳变


def run_pass(cands, n, reverse=False, init_conf=INIT_CONF):
    track, rejected = {}, {}
    px = py = None
    vx = vy = 0.0
    lost = 0
    order = range(n - 1, -1, -1) if reverse else range(n)
    for fi in order:
        cs = cands[fi] if fi < len(cands) else []
        if px is None:
            strong = [c for c in cs if c[4] >= init_conf]
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
    HERE = os.path.dirname(os.path.abspath(__file__))
    seg_ids = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4,5,6,7").split(",")]
    conf_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    init_conf = float(sys.argv[3]) if len(sys.argv) > 3 else INIT_CONF
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    for si in seg_ids:
        cp = os.path.join(HERE, "cand_s%d.json" % si)
        recs = json.load(open(cp, encoding="utf-8"))
        n_frames = max(r["f"] for r in recs) + 1
        cands = []
        for r in recs:
            cands.append([d for d in r["d"] if d[4] >= conf_min])

        tf, rf_ = run_pass(cands, n_frames, False, init_conf)
        tb, rb_ = run_pass(cands, n_frames, True, init_conf)
        track = dict(tf)
        for k, v in tb.items():
            if k not in track:
                track[k] = v
        # 长时轴跳变清理：并集缝合处按物理速度上限剔除低置信端
        while True:
            ks = sorted(track)
            bad = None
            for a, b in zip(ks, ks[1:]):
                if math.hypot(track[b][0] - track[a][0], track[b][1] - track[a][1]) > JUMP_HARD * (b - a):
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
                # 插值速度一致性：间隙内平均速度不得超过物理上限
                if math.hypot(tb2[0] - ta[0], tb2[1] - ta[1]) > JUMP_HARD * (b - a):
                    continue
                for j in range(a + 1, b):
                    u = (j - a) / (b - a)
                    e = u * u * (3 - 2 * u)
                    fills[j] = (ta[0] + (tb2[0] - ta[0]) * e, ta[1] + (tb2[1] - ta[1]) * e)

        suffix = "" if abs(init_conf - INIT_CONF) < 1e-9 else "_i%.2f" % init_conf
        out_path = os.path.join(HERE, "model_track_v2_s%d%s.mp4" % (si, suffix))
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (854, 356))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * fps))
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
