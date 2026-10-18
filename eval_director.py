# -*- coding: utf-8 -*-
"""【AI 判定】对导播输出做自动化客观判定（闭环里的"判卷"环节）

输入：auto_director.py 产出的 telemetry_s{si}.json
输出：逐项指标 + PASS/FAIL + 不达标时的归因提示

达标线（RESULT/段）：
  A 画面正确   aspect_ok            == 100%           画面绝不压扁/拉高
               in_bounds            == 100%           裁剪窗不出界
  B 球的物理   flash_jumps          <= 1              相邻帧位移超物理上限的次数
               static_lock_max      <= 20 帧          同一位置被"锁死"的最长帧数
               ball_cover           >= 45%            实检+插值覆盖
  C 人/owner   owner_cover          >= 90%            有球帧里判出带球人
               owner_switch_rate    <= 1.0 /s         不来回抖
               persons_ok           8 <= n <= 30      场上人数合理（10 球员+裁判）
  D 镜头质量   ball_in_view         >= 90%            镜头跟住了球
               pan_limit_ok         == 100%           摇镜速度不越限
               zoom_smooth          <= 0.02 倍/帧²    变焦无抽搐

用法: python eval_director.py 2[,3,6,10]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JUMP_LIMIT = 0.12       # "闪走"判定：实测球速上限 0.088/帧，0.12 以上只可能是链路错误而非真实运动
                        # （用户看到的真实闪走案例速度仅 0.047/帧，靠 static_lock 指标抓）
STATIC_TOL = 0.006      # 单帧位移小于此值视为"没动"
BOUNDS_TOL = 1e-3       # 遥测取整容差


def judge(seg_id):
    p = os.path.join(HERE, "telemetry_s%d.json" % seg_id)
    if not os.path.exists(p):
        raise SystemExit("missing %s, run auto_director.py %d first" % (p, seg_id))
    d = json.load(open(p, encoding="utf-8"))
    fr, n = d["frames"], d["n"]
    m = {}

    # A 画面
    m["aspect_ok"] = sum(1 for t in fr if abs(t["sx"] - t["sy"]) < 1e-6) / n
    m["in_bounds"] = sum(1 for t in fr
                         if -BOUNDS_TOL <= t["cx"] - t["cw"] / 854 / 2
                         and t["cx"] + t["cw"] / 854 / 2 <= 1 + BOUNDS_TOL
                         and -BOUNDS_TOL <= t["cy"] - t["ch"] / 356 / 2
                         and t["cy"] + t["ch"] / 356 / 2 <= 1 + BOUNDS_TOL) / n

    # B 球：闪走 / 静态锁 / 覆盖
    seq = [(t["f"], t["ball"], t["ball_kind"]) for t in fr if t["ball"]]
    jumps = 0
    for (f1, p1, _), (f2, p2, _) in zip(seq, seq[1:]):
        gap = max(1, f2 - f1)
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > JUMP_LIMIT * gap:
            jumps += 1
    m["flash_jumps"] = jumps
    # 静态锁 = **实检帧**在同一位置重复（插值段天然近似静止，不能算）
    det_seq = [(t["f"], t["ball"]) for t in fr if t["ball"] and t["ball_kind"] == "det"]
    run_len = cur = 1
    for (f1, p1), (f2, p2) in zip(det_seq, det_seq[1:]):
        if f2 - f1 <= 2 and math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < STATIC_TOL:
            cur += 1
        else:
            run_len, cur = max(run_len, cur), 1
    m["static_lock_max"] = max(run_len, cur) if det_seq else 0
    m["ball_cover"] = d["summary"]["ball_cover"]

    # C 人 / owner
    m["persons"] = d["summary"]["persons"]
    m["owner_cover"] = d["summary"]["owner_cover"]
    m["owner_switch_rate"] = d["summary"]["owner_switches"] / (n / d["fps"])

    # D 镜头
    inview = tot = 0
    for t in fr:
        if t["ball"] is None:
            continue
        tot += 1
        x, y = t["ball"]
        if (t["cx"] - t["cw"] / 854 / 2 <= x <= t["cx"] + t["cw"] / 854 / 2 and
                t["cy"] - t["ch"] / 356 / 2 <= y <= t["cy"] + t["ch"] / 356 / 2):
            inview += 1
    m["ball_in_view"] = inview / max(1, tot)
    m["pan_limit_ok"] = sum(1 for t in fr if t["pan_v"] <= t["vmax_pan"] + 1e-6) / n
    acc = [abs(fr[i + 1]["s"] - fr[i]["s"]) for i in range(len(fr) - 1)]
    m["zoom_jerk_p95"] = float(np.percentile(acc, 95)) if acc else 0.0
    m["zoom_jerk_max"] = max(acc) if acc else 0.0
    # 硬上限按本帧设计速率（含"急拉"提速）判定；典型值另设 0.02 防真实抽搐
    # s[i]→s[i+1] 的步长由第 i+1 帧的速率上限产生，判定必须用第 i+1 帧的上限
    m["zoom_cap_ok"] = sum(1 for i, a in enumerate(acc)
                           if a <= fr[i + 1]["vmax_zoom"] / d["fps"] + 5e-4) / max(1, len(acc))  # 遥测 s 保留4位，留取整容差

    checks = [
        ("A 画面比例锁定", m["aspect_ok"], "eq", 1.0),
        ("A 裁剪窗不出界", m["in_bounds"], "eq", 1.0),
        ("B 球无闪走", m["flash_jumps"], "le", 1),
        ("B 无静态误检锁", m["static_lock_max"], "le", 20),
        ("B 球覆盖", m["ball_cover"], "ge", 0.45),
        ("C 人数合理", m["persons"], "range", (8, 30)),
        ("C owner 覆盖", m["owner_cover"], "ge", 0.90),
        ("C owner 不抖", m["owner_switch_rate"], "le", 1.0),
        ("D 镜头跟住球", m["ball_in_view"], "ge", 0.90),
        ("D 摇镜不越速", m["pan_limit_ok"], "eq", 1.0),
        ("D 变焦典型平滑", m["zoom_jerk_p95"], "le", 0.02),
        ("D 变焦不越速", m["zoom_cap_ok"], "eq", 1.0),
    ]
    return seg_id, d["note"], m, checks


def fmt(v):
    return "%.3f" % v if isinstance(v, float) else str(v)


def main():
    ids = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2").split(",")]
    allpass = True
    for si in ids:
        si, note, m, checks = judge(si)
        print("\n===== seg %d  (%s)  AI 判定 =====" % (si, note))
        for name, val, op, thr in checks:
            ok = (abs(val - thr) < 1e-9 if op == "eq" else
                  val <= thr if op == "le" else
                  val >= thr if op == "ge" else
                  thr[0] <= val <= thr[1])
            allpass &= ok
            print("  [%s] %-16s %-8s %s %s" % (
                "PASS" if ok else "FAIL", name, fmt(val),
                {"eq": "==", "le": "<=", "ge": ">=", "range": "in"}[op], thr))
    print("\n总判定: %s" % ("全部达标" if allpass else "存在未达标项 → 需归因并迭代"))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
