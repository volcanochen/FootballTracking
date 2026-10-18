# -*- coding: utf-8 -*-
"""【编排器·两档版】模块1球 → 模块2人 → 模块3owner → 模块5两档镜头 → 视频 + 遥测

与 auto_director.py 的区别只在最后一环：镜头只有 FAR / NEAR 两档。
  FAR  —— 带球人在远半场：只取源画面 0.75×0.75 的区域，**等比放大填满全屏**
           （1/0.75 = 1.3333 倍，窗口与输出同宽高比 → 严格不变形）；
  NEAR —— 带球人在近半场：回到 s=1.0 的原始全幅，一点不放大。

用法: python auto_two_shot.py 2[,3,6,10]
"""
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mod_ball import track_ball, smooth_for_control      # noqa: E402
from mod_camera import crop                              # noqa: E402  （复用渲染）
from mod_owner import compute_owner                      # noqa: E402
from mod_person import load_persons                      # noqa: E402
from mod_two_shot import (FAR_WIN, S_FAR, TwoShotDirector,  # noqa: E402
                          calibrate, RAMP_SEC)

VIDEO = os.path.join(HERE, "full.mp4")
FPS = 29.97
SRC_W, SRC_H, OUT_W, OUT_H = 854, 356, 854, 356


def run(seg_id):
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    start, dur, note = segs[seg_id]
    f0, n = int(start * FPS), int(dur * FPS)
    print("=== seg %d  note=%s  frames=%d ===" % (seg_id, note, n), flush=True)

    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)

    # ---- 模块 1：球 ----
    ball_pos, ball_kind, bst = track_ball(seg_id, n)
    print("ball   [%s] cover=%.0f%%" % (bst["src"], 100 * bst["cover"]), flush=True)

    # ---- 模块 2：人 ----
    persons, per_frame = load_persons(seg_id, frames)
    print("person tracks=%d" % len(persons), flush=True)

    # ---- 模块 3：owner ----
    ctl_pos = smooth_for_control(ball_pos)
    owner_of, ost = compute_owner(ctl_pos, per_frame, persons, n)
    print("owner  %d/%d 有球帧 (%.0f%%) 切换=%.2f/s" % (
        ost["owner_frames"], ost["ball_frames"], 100 * ost["cover_vs_ball"],
        ost["switches_per_sec"]), flush=True)

    def box_of(f, pid):
        for (p, b) in per_frame.get(f, []):
            if p == pid:
                return b
        return None

    # ---- 标定：只用求半场线；远档倍率固定 1/0.75，不需要自适应 ----
    all_ys = []
    for f in range(n):
        for (pid, b) in per_frame.get(f, []):
            all_ys.append(float(b[3]))
    y_split = calibrate(all_ys)
    own_hs_far, own_hs_near = [], []
    for f in range(n):
        ob = box_of(f, owner_of.get(f)) if f in owner_of else None
        if ob is not None:
            (own_hs_far if ob[3] < y_split else own_hs_near).append((ob[3] - ob[1]) * SRC_H)
    print("标定   y_split=%.3f  远档倍率固定=%.4fx (=1/%.2f)" % (
        y_split, S_FAR, FAR_WIN), flush=True)
    print("       owner 身高：远半场 中位=%.0fpx → 输出 %.0fpx | 近半场 中位=%.0fpx" % (
        np.median(own_hs_far) if own_hs_far else 0,
        np.median(own_hs_far) * S_FAR if own_hs_far else 0,
        np.median(own_hs_near) if own_hs_near else 0), flush=True)

    # ---- 模块 5：两档镜头 + 渲染 ----
    d = TwoShotDirector(y_split, FPS, SRC_W, SRC_H, OUT_W, OUT_H)
    rng = np.random.default_rng(11)
    palette = {p["id"]: tuple(int(v) for v in rng.integers(60, 240, 3)) for p in persons}
    vw = cv2.VideoWriter(os.path.join(HERE, "two_shot_s%d.mp4" % seg_id),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    tele, trail, owner_px = [], [], []

    def to_out(p, view):
        return (int((p[0] * SRC_W - view["ox"]) * view["sx"]),
                int((p[1] * SRC_H - view["oy"]) * view["sy"]))

    for f in range(n):
        fr = frames[f]
        own = owner_of.get(f)
        own_box = box_of(f, own) if own is not None else None
        view = d.step(own_box, ctl_pos.get(f))
        out = crop(fr, view)

        # 半场线（远/近分界）：只在能看见的范围内画
        yl = to_out((0.0, y_split), view)[1]
        if -2 <= yl <= OUT_H + 2:
            for xx in range(0, OUT_W, 16):
                cv2.line(out, (xx, yl), (xx + 9, yl), (0, 255, 255), 1)
            cv2.putText(out, "halfway (far above / near below)", (8, yl - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)

        for (pid, b) in per_frame.get(f, []):
            p1, p2 = to_out((b[0], b[1]), view), to_out((b[2], b[3]), view)
            isown = (pid == own)
            col = (0, 220, 255) if isown else palette.get(pid, (200, 200, 200))
            cv2.rectangle(out, p1, p2, col, 2 if isown else 1)
            if isown:
                cv2.putText(out, "owner #%d" % pid, (p1[0], max(12, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        bp = ball_pos.get(f)
        if bp is not None:
            p = to_out(bp, view)
            kind, bconf = ball_kind[f]
            trail.append(p)
            if len(trail) > 20:
                trail.pop(0)
            for q in trail:
                cv2.circle(out, q, 2, (0, 200, 120), -1)
            if kind == "det":
                cv2.circle(out, p, 6, (0, 0, 255), 2)
                cv2.circle(out, p, 2, (0, 0, 255), -1)
                cv2.putText(out, "ball %.2f" % bconf, (p[0] + 8, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, p, 7, (0, 150, 255), 1)
                cv2.putText(out, "pred", (p[0] + 8, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 150, 255), 1, cv2.LINE_AA)
        else:
            trail.clear()

        far = (view["state"] == "FAR")
        col = (0, 255, 255) if far else (80, 220, 120)
        cv2.putText(out, "%s  zoom=%.2fx" % (view["state"], view["s"]), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        cv2.putText(out, "2-shot | far=%.4fx (1/%.2f) near=1.00x | ramp=%.1fs | t=%.1fs" % (
            S_FAR, FAR_WIN, RAMP_SEC, f / FPS), (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)

        mw, mh = OUT_W // 4, OUT_H // 4
        mm = cv2.resize(fr, (mw, mh))
        mm = cv2.copyMakeBorder(mm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(80, 80, 80))
        cv2.line(mm, (1, int(y_split * mh) + 1), (mw, int(y_split * mh) + 1),
                 (0, 255, 255), 1)
        cv2.rectangle(mm, (int(view["x0"] * mw) + 1, int(view["y0"] * mh) + 1),
                      (int(view["x0"] * mw) + 1 + int(view["cw"] / SRC_W * mw),
                       int(view["y0"] * mh) + 1 + int(view["ch"] / SRC_H * mh)),
                      (0, 255, 255), 1)
        if bp is not None:
            cv2.circle(mm, (int(bp[0] * mw) + 1, int(bp[1] * mh) + 1), 2, (0, 0, 255), -1)
        out[4:4 + mm.shape[0], OUT_W - mm.shape[1] - 4:OUT_W - 4] = mm
        vw.write(out)

        if own_box is not None:
            owner_px.append((own_box[3] - own_box[1]) * SRC_H * view["s"])
        tele.append({"f": f, "state": view["state"], "s": round(view["s"], 4),
                     "sx": round(view["sx"], 6), "sy": round(view["sy"], 6),
                     "cx": round(view["x0"] + view["cw"] / SRC_W / 2, 6),
                     "cy": round(view["y0"] + view["ch"] / SRC_H / 2, 6),
                     "cw": round(view["cw"], 4), "ch": round(view["ch"], 4),
                     "pan_v": view["pan_v"], "zoom_v": view["zoom_v"],
                     "vmax_pan": view["vmax_pan"], "vmax_zoom": view["vmax_zoom"],
                     "ball": (None if bp is None else [round(bp[0], 5), round(bp[1], 5)]),
                     "ball_kind": (None if bp is None else ball_kind[f][0]),
                     "owner": own,
                     "owner_box": (None if own_box is None else
                                   [round(float(v), 5) for v in own_box])})

    vw.release()
    json.dump({"seg": seg_id, "note": note, "n": n, "fps": FPS,
               "y_split": y_split, "s_far": S_FAR, "far_win": FAR_WIN,
               "anchor": view["anchor"] if False else __import__("mod_two_shot").ANCHOR,
               "ramp_sec": RAMP_SEC, "frames": tele,
               "summary": {"ball_cover": bst["cover"], "persons": len(persons),
                           "owner_cover": ost["cover_vs_ball"],
                           "owner_switches": ost["switches"]}},
              open(os.path.join(HERE, "telemetry_two_s%d.json" % seg_id), "w", encoding="utf-8"))
    print("shots: %s" % dict(Counter(t["state"] for t in tele)), flush=True)
    print("owner 输出像素高: 中位=%.0fpx" % (np.median(owner_px) if owner_px else 0), flush=True)
    print("saved two_shot_s%d.mp4 + telemetry_two_s%d.json" % (seg_id, seg_id), flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1] if len(sys.argv) > 1 else "2").split(","):
        run(int(s))
