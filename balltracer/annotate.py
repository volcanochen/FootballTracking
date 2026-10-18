# -*- coding: utf-8 -*-
"""标注视频绘制：红框=门限内检出 / 橙框pred=插值 / 灰虚线=双向均拒的突现检出 / 绿点=轨迹尾巴。"""
import json
import os

import cv2

from .config import OUT_W, OUT_H, CONF_MIN, FPS_FALLBACK


def load_cands(seg_id, cand_dir=".", conf_min=CONF_MIN):
    recs = json.load(open(os.path.join(cand_dir, "cand_s%d.json" % seg_id), encoding="utf-8"))
    n_frames = max(r["f"] for r in recs) + 1
    cands = [[d for d in r["d"] if d[4] >= conf_min] for r in recs]
    return cands, n_frames


def draw_video(seg_id, track, fills, rejected, video, seg_start_sec, fps, out_path, n_frames):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seg_start_sec * fps))
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    H, W = OUT_H, OUT_W
    trail = []

    def draw_box(x, y, w, h, color, thick, dashed=False):
        x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
        x2, y2 = x1 + max(5, int(w * W)), y1 + max(5, int(h * H))
        if not dashed:
            cv2.rectangle(cap_frame, (x1, y1), (x2, y2), color, thick)
            return
        for a, b2 in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                      ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            L = max(abs(b2[0] - a[0]), abs(b2[1] - a[1]))
            for t in range(0, L, 6):
                if a[1] == b2[1]:
                    p1 = (a[0] + t, a[1]); p2 = (min(a[0] + t + 3, b2[0]), a[1])
                else:
                    p1 = (a[0], a[1] + t); p2 = (a[0], min(a[1] + t + 3, b2[1]))
                cv2.line(cap_frame, p1, p2, color, 1)

    for fi in range(n_frames):
        ok, cap_frame = cap.read()
        if not ok:
            break
        drawn = False
        if fi in track:
            x, y, w, h, sc = track[fi]
            draw_box(x, y, w, h, (0, 0, 255), 2)
            cv2.putText(cap_frame, "ball %.2f" % sc, (int(x * W) + 6, int(y * H) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            trail.append((int(x * W), int(y * H)))
            drawn = True
        elif fi in fills:
            x, y = fills[fi]
            draw_box(x, y, 0.012, 0.02, (0, 150, 255), 1)
            cv2.putText(cap_frame, "pred", (int(x * W) + 6, int(y * H) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
            trail.append((int(x * W), int(y * H)))
            drawn = True
        if not drawn:
            for c in rejected.get(fi, []):
                draw_box(c[0], c[1], c[2], c[3], (160, 160, 160), 1, dashed=True)
        if len(trail) > 18:
            trail.pop(0)
        for ti in range(1, len(trail)):
            cv2.circle(cap_frame, trail[ti], 1, (0, 200, 120), -1)
        cv2.putText(cap_frame, "seg%d %.1fs" % (seg_id, fi / fps), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        wr.write(cap_frame)
    wr.release()
    cap.release()
