# -*- coding: utf-8 -*-
"""全段 Viterbi 全局轨迹优化。

贪心跟踪的问题：每帧只看上一帧，锁上球鞋/杂物后靠事后规则（静止断链、黑名单）补救，
且断链重选可能锁到另一个误检（如白线上的绿色反光点）。

Viterbi 思路：把整段所有帧的所有候选（青色候选 + YOLO 球框 + 「无球」虚态）
看作一个 HMM，全局求一条使总得分最大的路径：
  - 发射分：候选质量（球尺度面积 × 人框惩罚 × YOLO 印证）
  - 转移分：运动连续性（位移高斯惩罚 + 速度门限），「无球」态负责漏检区间
真球路径=长而连续的候选链；球鞋路径=静止但发射分被人框惩罚压死；
闪现误检=孤点，接进路径要付两次 Init 惩罚 → 全局被抛弃。

输出 dets_vit/<si>.json（≤15 帧缺口已插值），供 tracks.js BALL_SRC=vit 消费。
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 与 tracks.js / color_ball.py 对齐的参数 ----
AREA_LO, AREA_HI = 4, 45          # 球尺度面积（854x356）
GATE = 0.14                       # 单帧(0.1s)最大位移门限，超过禁止转移
SIGMA = 0.06                      # 位移惩罚尺度：cost=-(d/SIGMA)^2
P_MISS = 0.7                      # cand→ABS 转移惩罚（log 因子）
P_INIT = 1.0                      # ABS→cand 初始化惩罚
E_ABS = math.log(0.15)            # 「无球」虚态发射分
MAX_GAP = 15                      # 插值补洞上限（帧，0.1s/帧 → 1.5s）


def person_penalty(cand, people):
    """人框惩罚（配合运动证据）：
    - 框外 → 1.0
    - 近脚区（框底部 22%）→ 0.5：球和球鞋都在这，交给运动证据区分
    - 身体中部/上部（短裤/球衣）→ 0.05：绝不可能是球"""
    if not people:
        return 1.0
    for p in people:
        px, py = p["x"] - p["w"] / 2, p["y"] - p["h"] / 2
        if px <= cand["x"] <= px + p["w"] and py <= cand["y"] <= py + p["h"]:
            h = p["h"]
            feet_y = p["y"] + h / 2
            feet_x = p["x"]
            above_feet = feet_y - cand["y"]   # 距框底高度
            if above_feet < h * 0.22:
                return 0.5
            return 0.05
    return 1.0


def propagate_people(dets, win=2):
    """人框跨帧传播：第 i 帧用 [i-win, i+win] 内所有检出的人框取并集，
    填补 YOLO 漏检帧（16%）。球员 0.2s 内位移小于框宽，传播安全。"""
    n = len(dets)
    out = []
    for i in range(n):
        merged = []
        for j in range(max(0, i - win), min(n, i + win + 1)):
            for p in dets[j].get("people") or []:
                dup = False
                for m in merged:
                    ix = max(p["x"] - p["w"] / 2, m["x"] - m["w"] / 2)
                    iy = max(p["y"] - p["h"] / 2, m["y"] - m["h"] / 2)
                    x2 = min(p["x"] + p["w"] / 2, m["x"] + m["w"] / 2)
                    y2 = min(p["y"] + p["h"] / 2, m["y"] + m["h"] / 2)
                    inter = max(0, x2 - ix) * max(0, y2 - iy)
                    if inter > 0.5 * min(p["w"] * p["h"], m["w"] * m["h"]):
                        dup = True
                        break
                if not dup:
                    merged.append({"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"]})
        out.append(merged)
    return out


def mot_fac(mot):
    """运动证据 → 发射分乘数：静止物（牌/鞋/线）压到 0.15，mot≥6 给满"""
    if mot is None:
        return 0.6   # 段首无前帧可差分，取中性
    return min(1.0, 0.15 + 0.85 * min(1.0, mot / 6.0))


def facing_fac(cand, pose_ppl):
    """身体朝向先验：球在球员面朝的前方（多帧经验：控球人在球后）。
    - 找最近姿态人（质心距 <0.15）
    - faceConf 加权：前方 ×1.25，身后 ×0.7；无朝向信息 → 1.0"""
    if not pose_ppl:
        return 1.0
    best, best_d = None, 1e9
    for p in pose_ppl:
        d = ((cand["x"] - p["mx"]) ** 2 + (cand["y"] - p["my"]) ** 2) ** 0.5
        if d < best_d:
            best_d, best = d, p
    if best is None or best_d > 0.15 or best["faceConf"] < 0.15:
        return 1.0
    dx, dy = cand["x"] - best["mx"], cand["y"] - best["my"]
    r = (dx * dx + dy * dy) ** 0.5
    if r < 0.02:
        return 1.0   # 几乎在身体上，方向无意义
    front = (dx * best["fx"] + dy * best["fy"]) > 0
    base = 1.25 if front else 0.7
    return 1.0 + (base - 1.0) * best["faceConf"]


def build_candidates(rec, cy, ball_mot, people, pose_ppl=None):
    """一帧的所有候选状态：[{x, y, score}]（发射分含运动证据+人框惩罚+朝向先验）
    VIT_MODE=yolo：只用 YOLO 球框。
    固定青色光源 (0.630,0.565) 一律排除（已证实非球）。"""
    cands = []
    LIGHT = (0.630, 0.565)
    if os.environ.get("VIT_MODE") != "yolo":
        for c in (cy or []):
            if abs(c["x"] - LIGHT[0]) < 0.04 and abs(c["y"] - LIGHT[1]) < 0.04:
                continue
            base = 0.7 if AREA_LO <= c["area"] <= AREA_HI else 0.25
            s = base * person_penalty(c, people) * mot_fac(c.get("mot")) * facing_fac(c, pose_ppl)
            if s < 0.05:
                continue
            cands.append({"x": c["x"], "y": c["y"], "score": s})
    yb = rec.get("ball")
    if yb and not (abs(yb["x"] - LIGHT[0]) < 0.04 and abs(yb["y"] - LIGHT[1]) < 0.04):
        # YOLO 球框是主力候选（白/橙/青多种球色，YOLO 不挑色）。
        # 高分检出目检全是真球；低分（鞋误检）靠 人框惩罚+运动证据 压制。
        yc = {"x": yb["x"], "y": yb["y"]}
        s = (0.3 + 0.7 * min(1.0, yb["score"])) * person_penalty(yc, people) \
            * mot_fac(ball_mot) * facing_fac(yc, pose_ppl)
        cands.append({"x": yb["x"], "y": yb["y"], "score": s, "yolo": True})
        # 与某 cyan 候选重合 → 互相印证加分
        for c in cands:
            if not c.get("yolo") and abs(c["x"] - yb["x"]) < 0.03 and abs(c["y"] - yb["y"]) < 0.03:
                c["score"] = min(1.0, c["score"] + 0.35)
    return cands


def viterbi_segment(dets, cyans, mots, people_prop, poses=None):
    """返回每帧的选球结果 [{ball, real}]，ball=None 表示无球"""
    n = len(dets)
    cands_per = [build_candidates(dets[i],
                                  cyans[i]["cands"] if cyans and i < len(cyans) else None,
                                  mots[i].get("ballMot") if mots and i < len(mots) else None,
                                  people_prop[i],
                                  poses[i]["ppl"] if poses and i < len(poses) else None)
                 for i in range(n)]

    # dp[i][k]: k=-1 为 ABS，其余为候选下标
    dp = [[-1e18] * (len(cands_per[i]) + 1) for i in range(n)]
    bk = [[-1] * (len(cands_per[i]) + 1) for i in range(n)]

    def emit(c):
        return math.log(max(c["score"], 0.05)) if c else E_ABS

    for k in range(len(cands_per[0]) + 1):
        c = cands_per[0][k] if k < len(cands_per[0]) else None
        dp[0][k] = emit(c) - (P_INIT if c else 0.0)
        bk[0][k] = -2  # 起点标记

    for i in range(1, n):
        cur, prev = cands_per[i], cands_per[i - 1]
        nc, np_ = len(cur), len(prev)
        for k in range(nc + 1):
            c = cur[k] if k < nc else None
            best, arg = -1e18, -1
            if c is None:
                # ABS 态：从任意前态转移，代价只差 miss/init 惩罚；记得加 ABS 发射分
                for j in range(np_ + 1):
                    pc = prev[j] if j < np_ else None
                    if pc is None:
                        v = dp[i - 1][j] + E_ABS            # ABS→ABS
                    else:
                        v = dp[i - 1][j] - P_MISS + E_ABS   # cand→ABS
                    if v > best:
                        best, arg = v, j
            else:
                e = emit(c)
                for j in range(np_ + 1):
                    pc = prev[j] if j < np_ else None
                    if pc is None:
                        v = dp[i - 1][j] - P_INIT   # ABS→cand
                    else:
                        d = math.hypot(c["x"] - pc["x"], c["y"] - pc["y"])
                        if d > GATE:
                            continue                 # 禁止瞬移
                        v = dp[i - 1][j] - (d / SIGMA) ** 2
                    v += e
                    if v > best:
                        best, arg = v, j
            dp[i][k] = best
            bk[i][k] = arg

    # 回溯
    last = max(range(len(dp[n - 1])), key=lambda k: dp[n - 1][k])
    path = [last]
    for i in range(n - 1, 0, -1):
        path.append(bk[i][path[-1]])
    path.reverse()

    # 展开为球路径
    out = []
    for i, k in enumerate(path):
        c = cands_per[i][k] if k < len(cands_per[i]) else None
        out.append({"ball": ({"x": c["x"], "y": c["y"], "score": round(c["score"], 3)} if c else None),
                    "real": c is not None})
    return out


def fill_gaps(path):
    """≤MAX_GAP 帧的缺口做线性插值（与 tracks.js fillGaps 语义一致）"""
    n = len(path)
    i = 0
    while i < n:
        if path[i]["real"]:
            j = i + 1
            while j < n and path[j] and not path[j]["real"]:
                j += 1
            if j < n and path[j]["real"] and (j - i - 1) <= MAX_GAP:
                a, b = path[i]["ball"], path[j]["ball"]
                gap = j - i - 1
                for k in range(1, gap + 1):
                    u = k / (gap + 1)
                    ease = u * u * (3 - 2 * u)
                    path[i + k]["ball"] = {"x": a["x"] + (b["x"] - a["x"]) * ease,
                                           "y": a["y"] + (b["y"] - a["y"]) * ease,
                                           "score": 0.3}
                    path[i + k]["real"] = False
                    path[i + k]["interp"] = True
                i = j
            else:
                i = j if j > i else i + 1
        else:
            i += 1
    return path


def main():
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    only = [int(x) for x in os.sys.argv[1].split(",")] if len(os.sys.argv) > 1 else None
    os.makedirs(os.path.join(HERE, "dets_vit"), exist_ok=True)
    for si in range(len(segs)):
        if only is not None and si not in only:
            continue
        mp = os.path.join(HERE, "dets_mot", "%d.json" % si)
        if os.environ.get("VIT_DET") == "ft":
            dp_ = os.path.join(HERE, "dets_ft", "%d.json" % si)
            mp = os.path.join(HERE, "dets_mot_ft", "%d.json" % si)
        if not os.path.exists(dp_):
            continue
        dets = json.load(open(dp_, encoding="utf-8"))
        cyp = os.path.join(HERE, "dets_cyan", "%d.json" % si)
        cyans = json.load(open(cyp, encoding="utf-8")) if os.path.exists(cyp) else None
        mots = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None
        pp_ = os.path.join(HERE, "dets_pose", "%d.json" % si)
        poses = json.load(open(pp_, encoding="utf-8")) if os.path.exists(pp_) else None
        path = viterbi_segment(dets, cyans, mots, propagate_people(dets), poses)
        path = fill_gaps(path)
        # 对齐 dets 记录格式输出
        out = []
        for i, r in enumerate(dets):
            b = path[i]["ball"]
            out.append({"f": r["f"], "t": r["t"],
                        "ball": ({"x": b["x"], "y": b["y"], "w": 0.008, "h": 0.008,
                                  "score": b["score"]} if b else None),
                        "real": path[i]["real"], "interp": path[i].get("interp", False)})
        with open(os.path.join(HERE, "dets_vit", "%d.json" % si), "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False)
        n_real = sum(1 for r in out if r["real"])
        n_int = sum(1 for r in out if r.get("interp"))
        print("seg %d frames=%d real=%d(%.0f%%) interp=%d covered=%.0f%%" % (
            si, len(out), n_real, 100 * n_real / len(out), n_int,
            100 * (n_real + n_int) / len(out)), flush=True)
    print("done")


if __name__ == "__main__":
    main()
