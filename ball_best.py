# -*- coding: utf-8 -*-
"""【服务器·质量优先】球轨迹生成：三模型融合全量推理 + 双向时序门限 + 插值

与之前导播用的 cand_s{si}.json 的区别：
  - 旧缓存由 collect_candidates.py 以 MPEG 式 I/P 帧调度采集（P 帧只跑 1 个切片 + vb 全帧
    = 2 次推理），是给端侧省算力的**降级版**，候选池本就稀疏；
  - 本脚本**每帧 5 次推理**：切片模型 3 切片 @2x（等效分辨率 ~4.5x）+ vb 全帧 + old 全帧，
    conf 0.05 低阈高召回 —— 即评审闭环里验证过的 "best ensemble" 配置。

检测代码直接复用 anno_track_video_tile.py（保证与出片评审完全一致），
跟踪复用 balltracer.build_track（双向门限 + 跳变清理 + 插值）。

用法: python ball_best.py 2,3,6,10 [tile_weights] [conf=0.05]
输出: cand_best_<tag>_s{si}.json
"""
import json
import math
import os
import sys

import cv2

from anno_track_video_tile import tile_predict, full_predict   # 复用，杜绝实现漂移
from balltracer import build_track

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 29.97
MERGE_DIST = 0.02


def main():
    from ultralytics import YOLO
    ids = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2").split(",")]
    tile_w = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, "runs", "ball_tile_v3", "weights", "best.pt")
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(tile_w))))

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    m_tile = YOLO(tile_w)
    m_full = YOLO(os.path.join(HERE, "runs", "ball11n_vb", "weights", "best.pt"))
    m_old = YOLO(os.path.join(HERE, "runs", "ball11n", "weights", "best.pt"))
    print("tile=%s conf=%.2f" % (tag, conf), flush=True)

    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    for si in ids:
        out_path = os.path.join(HERE, "cand_best_%s_s%d.json" % (tag, si))
        start, dur, note = segs[si]
        n = int(dur * FPS)
        if os.path.exists(out_path):
            recs = json.load(open(out_path, encoding="utf-8"))
            cands = [r["d"] for r in recs]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * FPS))
            cands = []
            for fi in range(n):
                ok, fr = cap.read()
                if not ok:
                    break
                dets = tile_predict(m_tile, fr, conf) + full_predict(m_full, fr, conf) \
                    + full_predict(m_old, fr, conf)
                dets.sort(key=lambda d: -d[4])
                merged = []
                for d in dets:
                    if all(math.hypot(d[0] - k[0], d[1] - k[1]) > MERGE_DIST for k in merged):
                        merged.append(d)
                cands.append(merged)
                if fi % 100 == 0:
                    print("  seg%d frame %d/%d cands=%d" % (si, fi, n, len(merged)), flush=True)
            json.dump([{"f": i, "d": c} for i, c in enumerate(cands)],
                      open(out_path, "w", encoding="utf-8"))
            print("saved %s" % os.path.basename(out_path), flush=True)

        ncand_per_frame = sum(len(c) for c in cands) / max(1, len(cands))
        for ic in (0.08, 0.30):
            t, fl, rj = build_track(cands, len(cands), init_conf=ic)
            cov = (len(t) + len(fl)) / max(1, len(cands))
            print("  seg%d note=%-12s init=%.2f det=%d interp=%d cover=%.0f%% "
                  "(cands/frame=%.1f)" % (si, note, ic, len(t), len(fl), 100 * cov,
                                          ncand_per_frame), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
