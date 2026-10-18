# -*- coding: utf-8 -*-
"""时序门限跟踪：双向物理速度门限 + 长时轴跳变清理 + 插值补帧。

核心思想（来自 VCNC 跟焦需求）：场上只有一个球；高置信锁定后只接受相邻位置的候选；
高速/遮挡时单帧不可判 -> 不强求每帧检出，关键帧找对即可，中间按物理规律补轨迹。
"""
import math

from .config import (GATE_BASE, GATE_V, GATE_MAX, RECAP_STEP, MAXLOST, INIT_CONF,
                     FILL_MAX, JUMP_HARD)


def run_pass(cands, n, reverse=False, init_conf=INIT_CONF):
    """单向往返门限。返回 (track, rejected)。"""
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


def build_track(cands, n_frames, init_conf=INIT_CONF):
    """双向缝合 + 长时轴跳变清理 + 插值，返回 (track, fills, rejected)。"""
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
    # 双向均拒的突现检出（灰虚线标注）
    rejected = {}
    for k in (set(rf_) | set(rb_)):
        if k not in track and k in rf_ and k in rb_:
            common = [c for c in rf_[k] if c in rb_[k]]
            if common:
                rejected[k] = common
    # 插值补帧
    ks = sorted(track)
    fills = {}
    for a, b in zip(ks, ks[1:]):
        gap = b - a - 1
        if 0 < gap <= FILL_MAX:
            ta, tb2 = track[a], track[b]
            if math.hypot(tb2[0] - ta[0], tb2[1] - ta[1]) > JUMP_HARD * (b - a):
                continue
            for j in range(a + 1, b):
                u = (j - a) / (b - a)
                e = u * u * (3 - 2 * u)
                fills[j] = (ta[0] + (tb2[0] - ta[0]) * e, ta[1] + (tb2[1] - ta[1]) * e)
    return track, fills, rejected
