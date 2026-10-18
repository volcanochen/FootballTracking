# -*- coding: utf-8 -*-
"""运动证据 pass：固定机位下，用相邻采样帧（0.1s 间隔）的 |Δ(B−R)| 色度差分
给每个候选打运动分 mot：
  - 静态物（安全出口牌/静止球鞋/白线反光）→ mot ≈ 0
  - 移动的球 → mot ≈ 3~8
青色球亮度与草皮接近（灰度差分弱），但 B−R 色度对比强（球 B≫R，草皮 B≈R）。

输出：
  - dets_cyan/<si>.json 原地 enriched：cands[i].mot
  - dets_mot/<si>.json：YOLO 球框的 mot（[{f, t, ballMot}]）
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
RAD = 4   # 采样半径（854 宽下约 9px 窗口）


def main():
    only = [int(x) for x in os.sys.argv[1].split(",")] if len(os.sys.argv) > 1 else None
    os.makedirs(os.path.join(HERE, "dets_mot"), exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    for si in range(len(segs)):
        if only is not None and si not in only:
            continue
        dp_ = os.path.join(HERE, "dets", "%d.json" % si)
        cp = os.path.join(HERE, "dets_cyan", "%d.json" % si)
        if not os.path.exists(dp_):
            continue
        dets = json.load(open(dp_, encoding="utf-8"))
        cyans = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
        prev_chr = None
        ball_mot = []
        for i in range(len(dets)):
            # dets[i] ↔ 视频帧 3i：取该组第 1 帧，丢弃后 2 帧
            ok, fr = cap.read()
            if ok:
                for _ in range(2):
                    ok2, _ = cap.read()
                    if not ok2:
                        break
            if not ok:
                ball_mot.append({"f": dets[i]["f"], "t": dets[i]["t"], "ballMot": None})
                continue
            chr_ = fr[:, :, 0].astype(np.float32) - fr[:, :, 2].astype(np.float32)
            H, W = chr_.shape

            def mot_at(x, y):
                px, py = int(x * W), int(y * H)
                if px < RAD or py < RAD or px >= W - RAD or py >= H - RAD:
                    return None
                if prev_chr is None:
                    return None
                return float(np.abs(chr_[py - RAD:py + RAD + 1, px - RAD:px + RAD + 1]
                                    - prev_chr[py - RAD:py + RAD + 1, px - RAD:px + RAD + 1]).mean())

            rec = {"f": dets[i]["f"], "t": dets[i]["t"], "ballMot": None}
            yb = dets[i].get("ball")
            if yb:
                rec["ballMot"] = mot_at(yb["x"], yb["y"])
            ball_mot.append(rec)
            if cyans and i < len(cyans):
                for c in cyans[i]["cands"]:
                    c["mot"] = mot_at(c["x"], c["y"])
            prev_chr = chr_
        if cyans:
            with open(cp, "w", encoding="utf-8") as fp:
                json.dump(cyans, fp, ensure_ascii=False)
        with open(os.path.join(HERE, "dets_mot", "%d.json" % si), "w", encoding="utf-8") as fp:
            json.dump(ball_mot, fp, ensure_ascii=False)
        ms = [c["mot"] for r in (cyans or []) for c in r["cands"] if c.get("mot") is not None]
        print("seg %d done candsMot n=%d p50=%.2f p90=%.2f" % (
            si, len(ms), np.percentile(ms, 50) if ms else -1, np.percentile(ms, 90) if ms else -1), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
