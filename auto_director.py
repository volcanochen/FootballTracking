# -*- coding: utf-8 -*-
"""【编排器】把四个模块接起来，输出自动导播（虚拟摇镜头）视频 + 逐帧遥测

  模块1 球    mod_ball.track_ball        → 每帧球位（实检/插值）
  模块2 人    mod_person.load_persons    → 每人稳定 ID + 框
  模块3 owner mod_owner.compute_owner    → 每帧带球人
  模块4 镜头  mod_camera.Director        → 每帧裁剪窗（比例锁定 + 弹簧手感）

同时写出 telemetry_s{si}.json（逐帧状态/裁剪窗/球位/owner），供 eval_director.py 做 AI 判定。

用法: python auto_director.py 2[,3,6,10]
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mod_ball import track_ball, smooth_for_control    # noqa: E402
from mod_camera import Director, crop                  # noqa: E402
from vidout import H264Writer                            # noqa: E402
from mod_owner import compute_owner                    # noqa: E402
from mod_person import load_persons                    # noqa: E402

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
    print("ball   [%s] det=%d interp=%d cover=%.0f%%" % (
        bst["src"], bst["det"], bst["interp"], 100 * bst["cover"]), flush=True)

    # ---- 模块 2：人 ----
    persons, per_frame = load_persons(seg_id, frames)
    print("person tracks=%d" % len(persons), flush=True)

    # ---- 模块 3：owner（用去噪球位：控制层不该追检测噪点）----
    ctl_pos = smooth_for_control(ball_pos)
    owner_of, ost = compute_owner(ctl_pos, per_frame, persons, n)
    print("owner  %d/%d 有球帧 (%.0f%%) 切换=%.2f/s" % (
        ost["owner_frames"], ost["ball_frames"], 100 * ost["cover_vs_ball"],
        ost["switches_per_sec"]), flush=True)

    # ---- 球速度 + 自适应射门阈值（模块 4 的输入）----
    fs = sorted(ball_pos)
    ball_vel = {}
    for i, f in enumerate(fs):
        j = fs[max(0, i - 3)]
        ball_vel[f] = (np.array(ball_pos[f]) - np.array(ball_pos[j])) / max(1, f - j)
    shot = Director.adaptive_shot_speed(ball_vel)
    print("shot threshold=%.4f/frame" % shot, flush=True)

    # ---- 模块 4：导演 + 渲染 ----
    d = Director(shot, FPS, SRC_W, SRC_H, OUT_W, OUT_H)
    rng = np.random.default_rng(11)
    palette = {p["id"]: tuple(int(v) for v in rng.integers(60, 240, 3)) for p in persons}
    # H.264/yuv420p/faststart 写出，close() 内含容器自检（禁用 mp4v）
    vw = H264Writer(os.path.join(HERE, "auto_director_s%d.mp4" % seg_id),
                    OUT_W, OUT_H, FPS)
    tele, trail, zoom_log, owner_px = [], [], [], []

    def to_out(p, view):
        return (int((p[0] * SRC_W - view["ox"]) * view["sx"]),
                int((p[1] * SRC_H - view["oy"]) * view["sy"]))

    for f in range(n):
        fr = frames[f]
        bp = ball_pos.get(f)
        v = ball_vel.get(f, np.zeros(2))
        own = owner_of.get(f)
        own_box = None
        if own is not None:
            for (pid, b) in per_frame.get(f, []):
                if pid == own:
                    own_box = b
                    break
        view = d.step(ctl_pos.get(f), v, own_box)
        out = crop(fr, view)

        for (pid, b) in per_frame.get(f, []):
            p1, p2 = to_out((b[0], b[1]), view), to_out((b[2], b[3]), view)
            isown = (pid == own)
            col = (0, 220, 255) if isown else palette.get(pid, (200, 200, 200))
            cv2.rectangle(out, p1, p2, col, 2 if isown else 1)
            if isown:
                cv2.putText(out, "owner #%d" % pid, (p1[0], max(12, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        if bp is not None:
            p = to_out(bp, view)
            kind, bconf = ball_kind[f]
            trail.append(p)
            if len(trail) > 20:
                trail.pop(0)
            for q in trail:
                cv2.circle(out, q, 2, (0, 200, 120), -1)
            if kind == "det":                      # 实检：实心红点 + 置信
                cv2.circle(out, p, 6, (0, 0, 255), 2)
                cv2.circle(out, p, 2, (0, 0, 255), -1)
                cv2.putText(out, "ball %.2f" % bconf, (p[0] + 8, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1, cv2.LINE_AA)
            else:                                  # 插值补位：橙色空心圈 + pred
                cv2.circle(out, p, 7, (0, 150, 255), 1)
                cv2.putText(out, "pred", (p[0] + 8, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 150, 255), 1, cv2.LINE_AA)
        else:
            trail.clear()

        cv2.putText(out, "%s  zoom=%.2fx" % (view["state"], view["s"]), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "ball %s det=%d%% interp=%d%% | t=%.1fs" % (
            bst["src"], round(100.0 * bst["det"] / n), round(100.0 * bst["interp"] / n),
            f / FPS), (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        mw, mh = OUT_W // 4, OUT_H // 4
        mm = cv2.resize(fr, (mw, mh))
        mm = cv2.copyMakeBorder(mm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(80, 80, 80))
        cv2.rectangle(mm, (int(view["x0"] * mw) + 1, int(view["y0"] * mh) + 1),
                      (int(view["x0"] * mw) + 1 + int(view["cw"] / SRC_W * mw),
                       int(view["y0"] * mh) + 1 + int(view["ch"] / SRC_H * mh)),
                      (0, 255, 255), 1)
        if bp is not None:
            cv2.circle(mm, (int(bp[0] * mw) + 1, int(bp[1] * mh) + 1), 2, (0, 0, 255), -1)
        out[4:4 + mm.shape[0], OUT_W - mm.shape[1] - 4:OUT_W - 4] = mm
        vw.write(out)

        zoom_log.append(view["s"])
        if own_box is not None:
            owner_px.append((own_box[3] - own_box[1]) * SRC_H * view["s"])
        tele.append({"f": f, "state": view["state"], "s": round(view["s"], 4),
                     "sx": round(view["sx"], 6), "sy": round(view["sy"], 6),
                     "cx": round(view["x0"] + view["cw"] / SRC_W / 2, 6),
                     "cy": round(view["y0"] + view["ch"] / SRC_H / 2, 6),
                     "cw": round(view["cw"], 4), "ch": round(view["ch"], 4),
                     "pan_v": round(view["pan_v"], 5), "zoom_v": round(view["zoom_v"], 5),
                     "vmax_pan": round(view["vmax_pan"], 5), "vmax_zoom": round(view["vmax_zoom"], 5),
                     "reframing": view["reframing"],
                     "ball": (None if bp is None else [round(bp[0], 5), round(bp[1], 5)]),
                     "ball_kind": (None if bp is None else ball_kind[f][0]),
                     "owner": own})

    vinfo = vw.close()
    print("视频自检 %s: %s %s %s faststart=%s" % (
        os.path.basename(vw.path), vinfo["codec"], vinfo["pix_fmt"],
        vinfo["fourcc"], vinfo["faststart"]), flush=True)
    json.dump({"seg": seg_id, "note": note, "n": n, "fps": FPS,
               "ball_src": bst["src"], "shot_threshold": shot, "frames": tele,
               "summary": {"ball_det": bst["det"], "ball_interp": bst["interp"],
                           "ball_cover": bst["cover"], "persons": len(persons),
                           "owner_cover": ost["cover_vs_ball"],
                           "owner_switches": ost["switches"]}},
              open(os.path.join(HERE, "telemetry_s%d.json" % seg_id), "w", encoding="utf-8"))
    from collections import Counter
    print("states: %s" % dict(Counter(t["state"] for t in tele)), flush=True)
    print("zoom: %.2f~%.2f  owner 高度中位=%.0fpx" % (
        min(zoom_log), max(zoom_log), np.median(owner_px) if owner_px else 0), flush=True)
    print("saved auto_director_s%d.mp4 + telemetry_s%d.json" % (seg_id, seg_id), flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1] if len(sys.argv) > 1 else "2").split(","):
        run(int(s))
