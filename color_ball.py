# -*- coding: utf-8 -*-
"""青色球颜色检测器验证：对比 YOLO 检出，统计覆盖率与误检数"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")

# 高亮青色：hue 70~95, sat>45, val>130（从标注帧球心采样确定）
H_LO, H_HI = 68, 98
SAT_MIN, VAL_MIN = 45, 130
AREA_LO, AREA_HI = 2, 120   # 854x356 下球约 3~8px 直径 → 面积 5~50


def cyan_candidates(bgr, drop_watermark=True):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (H_LO, SAT_MIN, VAL_MIN), (H_HI, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H, W = bgr.shape[:2]
    if drop_watermark:
        mask[: int(H * 0.18), : int(W * 0.4)] = 0   # 左上角「VCNC bilibili」水印也是青色
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if AREA_LO <= area <= AREA_HI and w <= 18 and h <= 18:
            ar = w / max(1, h)
            if 0.4 <= ar <= 2.5:
                out.append({"x": (x + w / 2) / W, "y": (y + h / 2) / H,
                            "area": int(area), "score": min(1.0, area / 20.0)})
    return out


def main():
    segs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 2, 6]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tot_frames = tot_yolo = tot_cyan = tot_union = tot_match = multi_max = 0
    for si in only:
        dets = json.load(open(os.path.join(HERE, "dets", "%d.json" % si), encoding="utf-8"))
        start = segs[si][0]
        f0 = int(start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for r in dets:
            ok, fr = cap.read()
            if not ok:
                break
            if r["f"] % 3 != 0:
                continue
            cands = cyan_candidates(fr)
            yb = r["ball"]
            tot_frames += 1
            if yb:
                tot_yolo += 1
            if cands:
                tot_cyan += 1
                multi_max = max(multi_max, len(cands))
            # 融合判定：颜色候选与 YOLO 球距离 < 0.06 视为同一球
            hit_yolo = False
            if yb:
                for c in cands:
                    if abs(c["x"] - yb["x"]) < 0.06 and abs(c["y"] - yb["y"]) < 0.06:
                        hit_yolo = True
                        break
            if yb and cands:
                tot_union += 1
                if hit_yolo:
                    tot_match += 1
        print("seg %d done" % si, flush=True)
    print("frames=%d yoloHasBall=%.1f%% cyanHasBall=%.1f%% bothPresent=%d agree=%.1f%% maxMultiCyan=%d" % (
        tot_frames, 100 * tot_yolo / max(1, tot_frames), 100 * tot_cyan / max(1, tot_frames),
        tot_union, 100 * tot_match / max(1, tot_union), multi_max))
    cap.release()


if __name__ == "__main__":
    main()
