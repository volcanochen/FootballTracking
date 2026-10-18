# -*- coding: utf-8 -*-
"""端到端管线：候选采集 -> 时序跟踪 -> 标注视频。"""
import json
import os

import cv2

from .collect import collect_candidates
from .track import build_track
from .annotate import load_cands, draw_video
from .config import (CONF, CONF_MIN, INIT_CONF, DEFAULT_TILE_WEIGHTS, FPS_FALLBACK)


def _fps(video):
    cap = cv2.VideoCapture(video)
    f = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    cap.release()
    return f


def run(video, segs_file, seg_ids, tile_weights=DEFAULT_TILE_WEIGHTS,
        full_weights=None, old_weights=None, conf=CONF, key_every=3,
        conf_min=CONF_MIN, init_conf=INIT_CONF, out_dir=".", cand_dir=None):
    """运行完整管线，输出 model_track_v2_s{id}[_i{init}].mp4。"""
    segs = json.load(open(segs_file, encoding="utf-8"))
    if isinstance(seg_ids, str):
        seg_ids = [int(x) for x in seg_ids.split(",")]
    cand_dir = cand_dir or out_dir
    fps = _fps(video)

    collect_candidates(video, segs, seg_ids, tile_weights, full_weights, old_weights,
                      conf, key_every, cand_dir)

    stats = []
    for si in seg_ids:
        cands, n_frames = load_cands(si, cand_dir, conf_min)
        track, fills, rejected = build_track(cands, n_frames, init_conf)
        suffix = "" if abs(init_conf - INIT_CONF) < 1e-9 else "_i%.2f" % init_conf
        out_path = os.path.join(out_dir, "model_track_v2_s%d%s.mp4" % (si, suffix))
        draw_video(si, track, fills, rejected, video, segs[si][0], fps, out_path, n_frames)
        n_rej = sum(len(v) for v in rejected.values())
        covered = len(track) + len(fills)
        print("saved %s" % out_path, flush=True)
        print("  frames=%d det=%d(%.0f%%) interp=%d rejected_far=%d empty=%d" % (
            n_frames, len(track), 100.0 * len(track) / n_frames, len(fills),
            n_rej, n_frames - covered), flush=True)
        stats.append({"seg": si, "frames": n_frames, "det": len(track),
                      "interp": len(fills), "rejected_far": n_rej,
                      "empty": n_frames - covered, "video": out_path})
    return stats
