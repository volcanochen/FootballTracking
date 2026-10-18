# -*- coding: utf-8 -*-
"""【模块 3】带球人（owner）推断：球位 + 人框 → 每帧"谁在踢球"

输入只依赖模块 1（球位）与模块 2（人框），不碰视频像素，因此可独立回放/单测。

规则（三项，缺一不可）：
  1. 脚点距离：框底中点离球最近者胜出；
  2. 迟滞：当前 owner 的距离打 0.7 折，防止同色队友间来回抖；
  3. 保持 + 补位：球丢失或 owner 当帧漏检时，沿用最近已知框（≤6 帧）并保持 ≤HOLD 帧。

自测/单独运行: python mod_owner.py 2
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER_GATE = 0.055        # 球到"脚点"的距离门限（归一化）：门限内 = 明确带球人
NEAR_GATE = 0.12          # 兜底门限：球在飞行/带球人漏检时，取最近者作为镜头目标
OWNER_HOLD = 30           # owner 丢失后的保持帧数
FILL_FRAMES = 6           # 允许用最近已知框补位的最大帧数
HYST = 0.7                # 当前 owner 的距离折扣（迟滞）
SWITCH_MARGIN = 0.6       # 换人门槛：新目标必须比当前 owner 近 40% 以上才切换
SWITCH_DWELL = 8          # 换人还需连续 8 帧都更优（时序投票），杜绝 A→B→A 抖动


def foot_point(b):
    return np.array([(b[0] + b[2]) / 2, b[3]])      # 框底中点 ≈ 脚


def center_of(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def compute_owner(ball_pos, per_frame, persons, n_frames,
                  gate=OWNER_GATE, hold=OWNER_HOLD, fill=FILL_FRAMES,
                  near_gate=NEAR_GATE):
    """→ (owner_of, stats)

    owner_of[f] = 人的 track id。三级取值：
      owned   —— 脚点距离 < gate（明确带球）
      held    —— 球丢失/漏检时沿用上一个 owner（≤hold 帧）
      nearest —— 兜底：最近者（< near_gate），保证镜头永远有目标（球在飞行时也合理）
    """
    owner_of, cur_owner, held = {}, None, 0
    pend, pend_n = None, 0
    last_known = {p["id"]: None for p in persons}
    switches = 0
    for f in range(n_frames):
        for (pid, b) in per_frame.get(f, []):
            last_known[pid] = (f, b)
        act = []
        for p in persons:
            b = p["boxes"].get(f)
            if b is not None:
                act.append((p["id"], b))
                continue
            lk = last_known[p["id"]]
            if lk is not None and f - lk[0] <= fill:
                act.append((p["id"], lk[1]))        # 漏检补位（借鉴遮挡推理）
        if f not in ball_pos:
            if cur_owner is not None and held < hold:
                owner_of[f] = cur_owner
                held += 1
            continue
        bp = np.array(ball_pos[f])
        best, bestd, d_cur = None, 1e9, None
        for (pid, b) in act:
            d = float(np.linalg.norm(foot_point(b) - bp))
            if cur_owner is not None and pid == cur_owner:
                d_cur = d
                d *= HYST                               # 迟滞：偏好当前 owner
            if d < bestd:
                best, bestd = pid, d
        src = "owned" if (best is not None and bestd < gate * (1.6 if cur_owner == best else 1.0)) \
            else ("nearest" if (best is not None and bestd / HYST < near_gate) else None)
        if best is not None and cur_owner is not None and best != cur_owner \
                and d_cur is not None and bestd > SWITCH_MARGIN * d_cur * HYST:
            best, src = cur_owner, "keep"      # 不够明显更近 → 保持，防横跳
        if src is not None:
            if cur_owner is None:
                cur_owner, held, pend, pend_n = best, 0, None, 0
            elif best == cur_owner:
                held, pend, pend_n = 0, None, 0
            else:
                # 换人需连续 SWITCH_DWELL 帧都更优（时序投票），杜绝 A→B→A 抖动
                pend, pend_n = (best, pend_n + 1) if pend == best else (best, 1)
                if pend_n >= SWITCH_DWELL:
                    switches += 1
                    cur_owner, held, pend, pend_n = best, 0, None, 0
            owner_of[f] = cur_owner
        elif cur_owner is not None and held < hold:
            held += 1
            owner_of[f] = cur_owner
        else:
            cur_owner, pend, pend_n = None, None, 0
    # 分母只算"有球帧"：球丢失时保持的 owner 属于兜底行为，不能算进覆盖率（否则 >100%）
    n_ball = sum(1 for f in range(n_frames) if f in ball_pos)
    owner_on_ball = sum(1 for f in range(n_frames) if f in ball_pos and f in owner_of)
    stats = {"owner_frames": len(owner_of), "ball_frames": n_ball,
             "owner_on_ball": owner_on_ball,
             "cover_vs_ball": owner_on_ball / max(1, n_ball),
             "switches": switches,
             "switches_per_sec": switches / max(1e-6, n_frames / 29.97)}
    return owner_of, stats


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import json

    import cv2
    from mod_ball import track_ball
    from mod_person import load_persons
    si = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    start, dur, note = segs[si]
    f0, n = int(start * 29.97), int(dur * 29.97)
    cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    ball_pos, _, bst = track_ball(si, len(frames))
    persons, per_frame = load_persons(si, frames)
    owner_of, st = compute_owner(ball_pos, per_frame, persons, len(frames))
    print("seg%d  ball cover=%.0f%%  owner=%d/%d 有球帧 (%.0f%%)  切换=%d (%.2f/s)" % (
        si, 100 * bst["cover"], st["owner_frames"], st["ball_frames"],
        100 * st["cover_vs_ball"], st["switches"], st["switches_per_sec"]))
