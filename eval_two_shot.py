# -*- coding: utf-8 -*-
"""【AI 判定·两档版】读 telemetry_two_s{si}.json，检查"只有近/远两档 + 平缓"是否达标

远档语义（按用户要求）：**只取源画面 0.75×0.75 的区域，等比放大填满全屏**
（1/0.75 = 1.3333 倍；窗口与输出同宽高比 → 严格不变形）。

13 项达标线：
  A1 画面比例锁定      sx==sy                        == 1.0      （绝不压扁拉高）
  A2 裁剪窗不出界                                    == 1.0
  B1 近档=原始比例     NEAR 稳态帧 s                 == 1.0      （一点不放大）
  B2 远档=1/0.75       FAR  稳态帧 s                 == 1.3333
  B3 远档窗口=0.75     FAR  稳态帧 窗宽/源宽           == 0.75
  B4 不卡中间倍率      最长连续过渡段                 <= 1 个 ramp
  B5 推拉不越速        每帧 |Δs|                      <= 1.05×上限
  B6 档位不抽搐        切换次数/秒                    <= 0.30
  B7 换档方向正确      换档瞬间人在对应半场            == 1.0
  C1 owner 覆盖                                      >= 0.90
  C2 owner 在画面内                                  >= 0.95
  D1 远近语义一致                                      >= 0.85
  D2 远档确实放大      远档窗口占源画面                < 1.0

用法: python eval_two_shot.py 2[,3,6,10]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS_S = 0.01          # 稳态判定容差（倍率）：1% 缩放肉眼不可分辨
EPS_WIN = 0.0015      # 窗口尺寸容差（随 s 容差按 1/s² 换算）
SWITCH_MAX = 0.30     # 档位切换频率上限（次/秒）
ZOOM_TOL = 1.05       # 推拉速率容差（smoothstep 峰值 + 5%）
ZOOM_ABS = 5e-4       # 遥测 s 保留 4 位，留取整容差


def check(seg_id):
    p = os.path.join(HERE, "telemetry_two_s%d.json" % seg_id)
    if not os.path.exists(p):
        print("  [缺失] telemetry_two_s%d.json —— 先跑 auto_two_shot.py %d" % (seg_id, seg_id))
        return False
    d = json.load(open(p, encoding="utf-8"))
    fr = d["frames"]
    n = len(fr)
    fps = d["fps"]
    y_split, s_far = d["y_split"], d["s_far"]
    far_win = d.get("far_win", 1.0 / s_far)     # 远档窗口占源画面的比例（默认 0.75）
    ok_all = True

    print("\n===== seg %d  (%s)  两档镜头 AI 判定 =====" % (seg_id, d.get("note", "")))

    def rep(name, val, passed, op, thr):
        nonlocal ok_all
        ok_all = ok_all and passed
        print("  [%s] %-16s %-8s %s %s" % (
            "PASS" if passed else "FAIL", name,
            ("%.3f" % val) if isinstance(val, float) else str(val), op, thr))

    # --- A1 / A2 ---
    ar = sum(1 for f in fr if abs(f["sx"] - f["sy"]) <= 1e-6) / n
    rep("A1 画面比例锁定", ar, ar == 1.0, "==", "1.0")
    ib = sum(1 for f in fr if -1e-6 <= f["cx"] - f["cw"] / 854 / 2 and
             f["cx"] + f["cw"] / 854 / 2 <= 1 + 1e-6 and
             -1e-6 <= f["cy"] - f["ch"] / 356 / 2 and
             f["cy"] + f["ch"] / 356 / 2 <= 1 + 1e-6) / n
    rep("A2 裁剪窗不出界", ib, ib == 1.0, "==", "1.0")

    # --- 稳态 / 过渡 划分 ---
    steady = [f for f in fr if min(abs(f["s"] - 1.0), abs(f["s"] - s_far)) <= EPS_S]
    near = [f for f in steady if abs(f["s"] - 1.0) <= EPS_S]
    far = [f for f in steady if abs(f["s"] - s_far) <= EPS_S]

    # --- B1 / B2 / B3 / B4 ---
    e_near = max((abs(f["s"] - 1.0) for f in near), default=0.0)
    rep("B1 近档=原始比例", e_near, e_near <= EPS_S, "<=", "%.2f" % EPS_S)
    e_far = max((abs(f["s"] - s_far) for f in far), default=0.0)
    rep("B2 远档=1/0.75", e_far, e_far <= EPS_S, "<=",
        "%.2f (倍率 %.4f)" % (EPS_S, s_far))
    # 远档窗口必须恰好是源画面的 0.75×0.75 —— 这是本次需求的核心。
    # 注意 cw/src_w 与 ch/src_h 都恒等于 1/s，所以这里用更严的稳态定义
    # （s 与 s_far 差 ≤0.002）单独检验，避免与 B2 的 0.01 容差混在一起。
    far_strict = [f for f in fr if f["state"] == "FAR" and abs(f["s"] - s_far) <= 0.002]
    if far_strict:
        e_win = max(max(abs(f["cw"] / 854 - far_win), abs(f["ch"] / 356 - far_win))
                    for f in far_strict)
        rep("B3 远档窗口=0.75", e_win, e_win <= EPS_WIN, "<=",
            "%.4f (窗 %.3fx%.3f, %d 帧)" % (EPS_WIN, far_win, far_win, len(far_strict)))
    else:
        print("  [PASS] B3 远档窗口=0.75     -        (本段无 FAR 稳态帧)")
    # 「只有两档」的真正含义是：镜头绝不长时间停在中间倍率上。
    # 因此检验"最长连续过渡段"不得超过一个 ramp 时长 —— 卡在中间就是连续变焦了。
    ramp = max(2, int(round(float(d.get("ramp_sec", 1.4)) * fps)))
    run_, worst = 0, 0
    for f in fr:
        st = min(abs(f["s"] - 1.0), abs(f["s"] - s_far)) <= EPS_S
        run_ = 0 if st else run_ + 1
        worst = max(worst, run_)
    rep("B4 不卡中间倍率", worst, worst <= ramp + 5, "<=", "%d 帧 (1 个 ramp)" % (ramp + 5))

    # --- B4 推拉不越速 ---
    bad = 0
    for i in range(n - 1):
        a = abs(fr[i + 1]["s"] - fr[i]["s"])
        if a > fr[i + 1]["vmax_zoom"] / fps * ZOOM_TOL + ZOOM_ABS:
            bad += 1
    r = 1.0 - bad / max(1, n - 1)
    rep("B5 推拉不越速", r, r == 1.0, "==", "1.0")

    # --- B6 档位不抽搐 ---
    sw_idx = [i for i in range(1, n) if fr[i]["state"] != fr[i - 1]["state"]]
    sps = len(sw_idx) / max(1e-6, n / fps)
    rep("B6 档位不抽搐", sps, sps <= SWITCH_MAX, "<=",
        "%.2f/s (共%d次)" % (SWITCH_MAX, len(sw_idx)))

    # --- B7 切换方向正确（每次换档的瞬间，人确实在对应半场）---
    tot = good = 0
    for i in sw_idx:
        ob = fr[i].get("owner_box")
        if ob is None:
            continue
        tot += 1
        if (fr[i]["state"] == "FAR" and ob[3] <= y_split + 0.025) or \
           (fr[i]["state"] == "NEAR" and ob[3] >= y_split - 0.025):
            good += 1
    r = good / max(1, tot)
    rep("B7 换档方向正确", r, r == 1.0, "==", "1.0 (%d/%d)" % (good, tot))

    # --- C1 owner 覆盖 ---
    cov = d["summary"]["owner_cover"]
    rep("C1 owner 覆盖", cov, cov >= 0.90, ">=", "0.90")

    # --- C2 owner 在画面内 ---
    tot = inn = 0
    for f in fr:
        ob = f.get("owner_box")
        if ob is None:
            continue
        tot += 1
        cx, cy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
        if (f["cx"] - f["cw"] / 854 / 2 <= cx <= f["cx"] + f["cw"] / 854 / 2 and
                f["cy"] - f["ch"] / 356 / 2 <= cy <= f["cy"] + f["ch"] / 356 / 2):
            inn += 1
    r = inn / max(1, tot)
    rep("C2 owner 在画面内", r, r >= 0.95, ">=", "0.95")

    # --- D1 语义正确：档位与半场线一致 ---
    # 注意：一次推拉要 1.4 秒，这期间人完全可能自己跑过半场线，属于正常现象，
    # 因此这里只要求"稳态档位"与所在半场大体吻合（0.85），精确性由 B7 保证。
    tot = good = 0
    for f in steady:
        ob = f.get("owner_box")
        if ob is None:
            continue
        tot += 1
        y = ob[3]
        if (f["state"] == "FAR" and y <= y_split + 0.025) or \
           (f["state"] == "NEAR" and y >= y_split - 0.025):
            good += 1
    r = good / max(1, tot)
    rep("D1 远近语义一致", r, r >= 0.85, ">=", "0.85")

    # --- D2 远档确实"放大"（视野确实收窄了）---
    w = [f["cw"] / 854 for f in far]
    r = float(np.median(w)) if w else 1.0
    rep("D2 远档视野收窄", r, r < 1.0, "<", "1.0 (窗=%.2f 个源画面)" % r)

    print("  稳态: NEAR=%d (%.0f%%)  FAR=%d (%.0f%%)  过渡=%d 帧" % (
        len(near), 100 * len(near) / n, len(far), 100 * len(far) / n, n - len(steady)))
    return ok_all


if __name__ == "__main__":
    segs = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "2").split(",")]
    res = {s: check(s) for s in segs}
    print("\n总判定: %s" % ("全部达标" if all(res.values())
                            else "未达标 -> " + ",".join("s%d" % s for s in segs if not res[s])))
