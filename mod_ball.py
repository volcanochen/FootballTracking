# -*- coding: utf-8 -*-
"""【模块 1】球感知：三模型融合候选 → 双向时序门限跟踪 → 插值补帧

对外只暴露一个函数 `track_ball()`，输入段号与帧数，输出每帧球位及其来源（实检/插值）。
检测与跟踪实现全部复用 balltracer（与评审闭环同一套代码，杜绝实现漂移）。

- 候选优先取 `cand_best_<tag>_s{si}.json`（每帧 5 次推理的质量优先融合，见 ball_best.py）；
- 缺失时退回 `cand_s{si}.json`（MPEG 式 I/P 帧降级缓存，端侧省算力用，精度较低）。

自测/单独运行: python mod_ball.py 2
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BALL_TAG = os.environ.get("BALL_TAG", "ball_tile_v3")
BALL_INIT = 0.08          # 降阈冷启动：远景弱信号也能起轨（宁可多起，由门限后级筛）

from balltracer import load_cands, build_track   # noqa: E402


def load_candidates(seg_id, tag=BALL_TAG, here=HERE):
    """返回 (cands, src_name)。cands[f] = [[x, y, w, h, conf], ...]（归一化坐标）"""
    best = os.path.join(here, "cand_best_%s_s%d.json" % (tag, seg_id))
    if os.path.exists(best):
        recs = json.load(open(best, encoding="utf-8"))
        return [r["d"] for r in recs], "best-ens/%s" % tag
    return load_cands(seg_id, here)[0], "legacy-IP-cache"


def track_ball(seg_id, n_frames, init_conf=BALL_INIT, tag=BALL_TAG, here=HERE):
    """→ (ball_pos, ball_kind, stats)

    ball_pos[f]  = (x, y)                归一化坐标
    ball_kind[f] = ("det", conf) | ("interp", 0.0)
    stats        = dict(源, 实检帧数, 插值帧数, 覆盖率)
    """
    cands, src = load_candidates(seg_id, tag, here)
    track, fills, _ = build_track(cands, min(n_frames, len(cands)), init_conf=init_conf)
    ball_pos, ball_kind = {}, {}
    for f in range(n_frames):
        if f in track:
            ball_pos[f] = (track[f][0], track[f][1])
            ball_kind[f] = ("det", float(track[f][4]))
        elif f in fills:
            ball_pos[f] = (fills[f][0], fills[f][1])
            ball_kind[f] = ("interp", 0.0)
    ndet = sum(1 for k in ball_kind.values() if k[0] == "det")
    stats = {"src": src, "det": ndet, "interp": len(fills),
             "cover": (ndet + len(fills)) / max(1, n_frames)}
    return ball_pos, ball_kind, stats


def smooth_for_control(ball_pos, win=3):
    """给"控制模块"用的去噪球位：3 帧中值滤波。

    低阈值重采后候选变密，球位偶发跳点（噪声）。镜头和 owner 若直接追噪点会导致
    摇镜跟不上、owner 来回换人。注意：**仅用于控制与 owner 推断，画面上仍画原始球位**，
    不掩盖真实检出精度。
    """
    out = {}
    ks = sorted(ball_pos)
    for i, f in enumerate(ks):
        nb = [ball_pos[g] for g in ks[max(0, i - win // 2): i + win // 2 + 1]]
        out[f] = (float(np.median([p[0] for p in nb])), float(np.median([p[1] for p in nb])))
    return out


if __name__ == "__main__":
    si = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    nf = int(segs[si][1] * 29.97)
    pos, kind, st = track_ball(si, nf)
    print("seg%d  src=%s  det=%d interp=%d cover=%.0f%%" % (
        si, st["src"], st["det"], st["interp"], 100 * st["cover"]))
