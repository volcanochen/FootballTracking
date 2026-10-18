# -*- coding: utf-8 -*-
"""基于候选缓存的锚点召回评测：单帧 vs 长时轴邻帧融合。
用法: eval_cache_anchors.py [mode=single|temporal] [R=0.03] [rad=1]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    rad = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    tot = hit = 0
    per = {}
    for si in range(8):
        dp = os.path.join(HERE, "dets_track", "%d.json" % si)
        cp = os.path.join(HERE, "cand_s%d.json" % si)
        if not (os.path.exists(dp) and os.path.exists(cp)):
            continue
        d = json.load(open(dp, encoding="utf-8"))
        cand = json.load(open(cp, encoding="utf-8"))
        st = [0, 0]
        for a, rec in enumerate(d):
            b = rec.get("ball")
            if not b or not rec.get("real"):
                continue
            fi = a * 3
            st[1] += 1
            if mode == "single":
                pool = [(d[0], d[1], d[4]) for d in (cand[fi]["d"] if fi < len(cand) else [])]
            else:
                pool = []
                for off in range(-rad, rad + 1):
                    j = fi + off
                    if 0 <= j < len(cand):
                        w = 1.0 if off == 0 else 0.6
                        pool += [(x, y, sc * w) for x, y, _, _, sc in cand[j]["d"]]
            best = min((math.hypot(x - b["x"], y - b["y"]) for x, y, sc in pool), default=9)
            if best < R:
                st[0] += 1
        per[si] = st
        tot += st[1]
        hit += st[0]
    for k in sorted(per):
        print("seg%d: %d/%d (%.0f%%)" % (k, per[k][0], per[k][1], 100.0 * per[k][0] / max(1, per[k][1])))
    print("%s R=%.3f recall: %d/%d (%.1f%%)" % (mode.upper(), R, hit, tot, 100.0 * hit / max(1, tot)))


if __name__ == "__main__":
    main()
