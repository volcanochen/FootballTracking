# -*- coding: utf-8 -*-
"""时序门限跟踪：双向物理速度门限 + 长时轴跳变清理 + 插值补帧。

核心思想（来自 VCNC 跟焦需求）：场上只有一个球；高置信锁定后只接受相邻位置的候选；
高速/遮挡时单帧不可判 -> 不强求每帧检出，关键帧找对即可，中间按物理规律补轨迹。
"""
import math

from .config import (GATE_BASE, GATE_V, GATE_MAX, RECAP_STEP, MAXLOST, INIT_CONF,
                     FILL_MAX, JUMP_HARD, SIZE_MIN, STATIC_STRICT_CONF)


def _size_ok(c):
    """球在源画面里是 20px 量级的小目标；实测 conf≥0.40 的候选 w 全部 ≥0.015，
    而白衣/纹理类误检普遍更小 → 尺寸下限是零成本的高精度过滤器。"""
    return c[2] >= SIZE_MIN and c[3] >= SIZE_MIN


def static_mask(cands, win=15, kmin=10, tol=0.010):
    """持久静止候选标记（疑似草皮纹理/场边物体误检）。

    判据：位置在 ±win 帧窗口内有 ≥kmin 帧出现于 tol 邻域内。真球不会长期钉在
    同一像素（要么被踢走要么被遮挡），满足该条件的几乎全是静态误检。
    """
    n = len(cands)
    static = [[False] * len(cs) for cs in cands]
    for f, cs in enumerate(cands):
        for i, c in enumerate(cs):
            x, y = c[0], c[1]
            cnt = 0
            for g in range(max(0, f - win), min(n, f + win + 1)):
                if g == f:
                    continue
                for d in cands[g]:
                    if abs(d[0] - x) <= tol and abs(d[1] - y) <= tol:
                        cnt += 1
                        break
            if cnt >= kmin:
                static[f][i] = True
    return static


def run_pass(cands, n, reverse=False, init_conf=INIT_CONF, static=None):
    """单向往返门限。返回 (track, rejected)。

    宁缺勿错（v4 哲学）+ 两类误检硬过滤：
      - 尺寸下限：w/h < SIZE_MIN 的候选直接丢弃（实测 conf≥0.40 的球从不这么小，
        白衣/纹理误检普遍更小——曾把链锁死在守门员上衣上几十帧）；
      - 持久静止候选：位置长期不动的低置信候选（草皮/场边物体）禁用。
        真球静止且清晰（conf ≥ STATIC_STRICT_CONF）时仍允许，不影响开球定位。
    """
    track, rejected = {}, {}
    px = py = None
    vx = vy = 0.0
    lost = 0
    order = range(n - 1, -1, -1) if reverse else range(n)
    for fi in order:
        cs_all = cands[fi] if fi < len(cands) else []
        sm = static[fi] if static is not None and fi < len(static) else None
        cs = []
        for j, c in enumerate(cs_all):
            if not _size_ok(c):
                continue
            if sm and sm[j] and c[4] < STATIC_STRICT_CONF:
                continue
            cs.append(c)
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


def build_track(cands, n_frames, init_conf=INIT_CONF, block_static=True):
    """双向缝合 + 长时轴跳变清理 + 静止误检清理 + 插值，返回 (track, fills, rejected)。"""
    static = static_mask(cands) if block_static else None
    tf, rf_ = run_pass(cands, n_frames, False, init_conf, static)
    tb, rb_ = run_pass(cands, n_frames, True, init_conf, static)
    track = dict(tf)
    for k, v in tb.items():
        if k not in track:
            track[k] = v
    # 静止误检清理：双向缝合后同一位置持续 ≥STAT_RUN 帧 → 判为静态 FP 锁，整段剔除。
    # 真球不会在比赛中十几秒钉在同一像素；被剔后插值也不会再桥接到误检上。
    ks = sorted(track)
    STAT_TOL, STAT_RUN = 0.006, 20
    runs, cur = [], []
    for a, b in zip(ks, ks[1:]):
        if not cur:
            cur = [a]
        if b - a <= 2 and math.hypot(track[b][0] - track[a][0],
                                     track[b][1] - track[a][1]) < STAT_TOL:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    if cur:
        runs.append(cur)
    for r in runs:
        if len(r) >= STAT_RUN:
            for k in r:
                del track[k]
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
