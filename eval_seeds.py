# -*- coding: utf-8 -*-
"""种子召回评测：在人工确认的球位（seeds_v2.json 过滤掉评审坏框）上，
统计模型单帧检出召回率。判据：检出框中心距种子点 < 0.015（归一化）。
用法: python eval_seeds.py <weights> [conf=0.10]
"""
import json
import os
import sys

import cv2


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    weights = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10

    seeds = json.load(open(os.path.join(HERE, "seeds_v2.json"), encoding="utf-8"))
    rv = json.load(open(os.path.join(HERE, "review_result.json"), encoding="utf-8"))
    bad = set(rv.get("badBoxes", []))
    good = []
    for s in seeds:
        if "s%d_i%04d#S0" % (s["seg"], s["i"]) in bad:
            continue
        good.append(s)
    print("seeds total=%d good=%d" % (len(seeds), len(good)))

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    model = YOLO(weights)
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    R = 0.015
    hit = hit_hi = 0
    miss = []
    by_seg = {}
    for s in good:
        seg = s["seg"]
        fi = int(segs[seg][0] * fps) + s["i"] * 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
        best = None  # (dist, conf) 距种子最近的检出
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                continue
            x, y = float(b.xywhn[0][0]), float(b.xywhn[0][1])
            dist = ((x - s["x"]) ** 2 + (y - s["y"]) ** 2) ** 0.5
            sc = float(b.conf[0])
            if best is None or dist < best[0]:
                best = (dist, sc)
        st = by_seg.setdefault(seg, [0, 0])
        st[1] += 1
        if best and best[0] < R:
            hit += 1
            st[0] += 1
            if best[1] > 0.5:
                hit_hi += 1
        else:
            miss.append("s%d_i%d(dist=%s)" % (seg, s["i"],
                         round(best[0], 3) if best else "none"))
    cap.release()
    print("weights=%s conf=%.2f" % (os.path.basename(weights), conf))
    print("recall=%d/%d (%.1f%%)  hiConf(>0.5)=%d" % (hit, len(good), 100.0 * hit / max(1, len(good)), hit_hi))
    for k in sorted(by_seg):
        print("  seg%d: %d/%d" % (k, by_seg[k][0], by_seg[k][1]))
    print("miss:", "; ".join(miss))


if __name__ == "__main__":
    main()
