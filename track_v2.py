# -*- coding: utf-8 -*-
"""track_v2: 锚点种子局部跟踪（替代 Viterbi 全局寻优）。

用户确认的正确球位置(seeds_v2 - 坏种子)作硬锚点；
锚点之间/前后只做"相邻候选"局部链式跟踪：
  - 预测位置 = 上一点 + 速度(EMA)，门限 R 随速度自适应（球不会瞬移）
  - 候选 = ft 微调模型检出(优先) + 青色运动候选(需 mot、排除固定光源)
  - 连续 MAXLOST 步找不到 → 断链留空（宁缺勿错）
  - 中等空洞由 make_anno_video 的 ≤6 锚点线性插值补（物理规律近似）
输出 dets_track/<si>.json（与 dets_vit 同 schema），供 make_anno_video 使用。

用法:
  python track_v2.py            # 全段跟踪 + 留一法验证
  python track_v2.py 0,3        # 只跑指定段
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIGHT = (0.630, 0.565)          # 固定青色光源（必排除）
LIGHT_R = 0.045
MAXLOST = 12                    # 连续丢 1.2s 断链
RECAP_STEP = 0.006              # 丢步后每步扩大的再捕获半径
RECAP_MAX = 0.05
FILL_MAX = 12                   # ≤1.2s 的空洞做插值补齐
STRONG_FT = 0.30                # ft 检出达到此分算强候选
WEAK_RUN_MAX = 8                # 连续弱候选超过此数断链（防骑球袜/衣物）
CYAN_MOT_HI = 10                # cyan mot 上限（球 3~8；衣物/人腿 16~46）
CYAN_AREA_HI = 40               # cyan 面积上限（球尺度）
MEET_R = 0.05                   # 前向/后向链汇合判定
GATE_BASE = 0.035               # 静止时基础门限
GATE_V = 2.0                    # 门限随速度放大系数
GATE_MAX = 0.10
FT_BONUS = 0.018                # ft 检出优先于 cyan
MOT_MIN_CYAN = 0.40             # cyan 候选最低运动证据
MOT_MIN_FT = 0.0                # 锚定区间内 ft 不设 mot 门槛
MOT_MIN_FT_FREE = 0.6           # 无锚区自由链的 ft mot 门槛
BALL_ALL_MIN = 0.15             # ballAll 纳入候选的最低分


def load_anchors():
    sv = json.load(open(os.path.join(HERE, "seeds_v2.json"), encoding="utf-8"))
    bad_idx = {29, 37, 57, 60, 62, 63, 64}
    rr = json.load(open(os.path.join(HERE, "review_result.json"), encoding="utf-8"))
    user_bad = {b.split("#")[0] for b in rr["badBoxes"] if b.endswith("#S0")}
    anch = {}
    for k, s in enumerate(sv):
        if k in bad_idx:
            continue
        if "s%d_i%04d" % (s["seg"], s["i"]) in user_bad:
            continue
        anch.setdefault(s["seg"], {})[s["i"]] = (s["x"], s["y"])
    return anch


def build_cands(dets, cyans, mots):
    """每锚点下标的候选列表 [{x,y,score,src,mot,tier}]；tier: strong/weak"""
    out = []
    for i, r in enumerate(dets):
        cs = []
        bm = (mots or [{}]*len(dets))[i].get("ballMot") if mots and i < len(mots) else None
        if r.get("ball"):
            cs.append({"x": r["ball"]["x"], "y": r["ball"]["y"],
                       "score": r["ball"].get("score", 0.5), "src": "ft",
                       "mot": bm if bm is not None else 1.0})
        for b in (r.get("ballAll") or []):
            if b.get("score", 0) >= BALL_ALL_MIN:
                cs.append({"x": b["x"], "y": b["y"], "score": b["score"],
                           "src": "ft", "mot": bm if bm is not None else 1.0})
        if cyans and i < len(cyans):
            for c in cyans[i].get("cands", []):
                if math.hypot(c["x"] - LIGHT[0], c["y"] - LIGHT[1]) < LIGHT_R:
                    continue
                mot = c.get("mot") or 0
                if mot < MOT_MIN_CYAN or mot > CYAN_MOT_HI:
                    continue
                if c.get("area", 99) > CYAN_AREA_HI:
                    continue
                cs.append({"x": c["x"], "y": c["y"], "score": c.get("score", 0.3),
                           "src": "cyan", "mot": mot})
        for c in cs:
            if c["src"] == "ft":
                c["tier"] = "strong" if c["score"] >= STRONG_FT else "weak"
            else:
                c["tier"] = "weak"      # cyan 一律弱候选（球袜/短裤污染源）
        out.append(cs)
    return out


def gate(v):
    return min(GATE_MAX, GATE_BASE + GATE_V * math.hypot(*v)) if any(v) else 0.05


def pick(cands, px, py, v, ft_mot_min, lost=0):
    """门限内选最优候选；返回 (cand, tier) 或 None。lost 越大门限越宽（再捕获）
    优先强候选；仅弱候选时也选（调用方计数弱链长度）"""
    R = min(gate(v) + RECAP_STEP * lost, gate(v) + RECAP_MAX)
    best, bc, btier = None, 1e9, None
    for c in cands:
        if c["src"] == "ft" and c["mot"] is not None and c["mot"] < ft_mot_min:
            continue
        d = math.hypot(c["x"] - px, c["y"] - py)
        if d > R:
            continue
        cost = d - (FT_BONUS if c["src"] == "ft" else 0.0) \
               - (0.008 * min(c.get("mot", 0) / 3.0, 1.0) if c["src"] == "cyan" else 0.0) \
               - (0.02 if c["tier"] == "strong" else 0.0)
        if cost < bc:
            bc, best, btier = cost, c, c["tier"]
    if best is None:
        return None
    return best, btier


def chain(cands, i0, p0, i_end, step, v0, ft_mot_min):
    """从 (i0,p0) 向 i_end 方向逐锚点步进。返回 {i:(x,y,tier)}（不含 i0）。
    连续弱候选超过 WEAK_RUN_MAX 断链（防骑球袜/衣物）。"""
    out = {}
    last = p0
    v = list(v0)
    lost = 0
    wrun = 0
    i = i0 + step
    while (step > 0 and i < i_end) or (step < 0 and i > i_end):
        px, py = last[0] + v[0], last[1] + v[1]
        got = pick(cands[i], px, py, v, ft_mot_min, lost) if i < len(cands) else None
        if got:
            c, tier = got
            out[i] = (c["x"], c["y"], tier)
            v = [0.65 * v[0] + 0.35 * (c["x"] - last[0]),
                 0.65 * v[1] + 0.35 * (c["y"] - last[1])]
            last = (c["x"], c["y"])
            lost = 0
            wrun = 0 if tier == "strong" else wrun + 1
            if wrun > WEAK_RUN_MAX:
                break
        else:
            v = [v[0] * 0.8, v[1] * 0.8]   # 摩擦衰减，预测点不飞
            lost += 1
            wrun += 1
            if lost > MAXLOST or wrun > WEAK_RUN_MAX:
                break
        i += step
    return out


def init_v(anch_seg, i0):
    """用锚点邻帧估计初速度（若无邻锚点则 0）"""
    for di in (1, -1, 2, -2):
        j = i0 + di
        if j in anch_seg:
            dt = abs(di)
            return [(anch_seg[j][0] - anch_seg[i0][0]) / dt / 1.0,
                    (anch_seg[j][1] - anch_seg[i0][1]) / dt / 1.0]
    return [0.0, 0.0]


def track_seg(anch_seg, cands, n):
    """整段跟踪，返回 pos: dict i->(x,y,src)；src: A=锚点/F=前向/B=后向"""
    pos = {}
    ks = sorted(anch_seg)
    if not ks:
        return pos, "free"      # 无锚段由调用方特殊处理
    for k in ks:
        pos[k] = (anch_seg[k][0], anch_seg[k][1], "A")
    # 相邻锚点对之间双向链+汇合
    for a, b in zip(ks, ks[1:]):
        if b - a <= 1:
            continue
        v0 = init_v(anch_seg, a)
        v1 = init_v(anch_seg, b)
        F = chain(cands, a, anch_seg[a], b, +1, v0, MOT_MIN_FT)
        Bk = chain(cands, b, anch_seg[b], a, -1, v1, MOT_MIN_FT)
        meet = None
        for j in sorted(F):
            if j in Bk and math.hypot(F[j][0] - Bk[j][0], F[j][1] - Bk[j][1]) < MEET_R:
                meet = j
                break
        if meet is not None:
            for j in range(a + 1, meet):
                if j in F:
                    pos[j] = (F[j][0], F[j][1], "w" if F[j][2] == "weak" else "F")
            for j in range(meet, b):
                if j in Bk:
                    pos[j] = (Bk[j][0], Bk[j][1], "w" if Bk[j][2] == "weak" else "B")
        else:
            for j, p in F.items():
                if a < j < b and (j not in pos):
                    pos[j] = (p[0], p[1], "w" if p[2] == "weak" else "F")
            for j, p in Bk.items():
                if a < j < b and (j not in pos):
                    pos[j] = (p[0], p[1], "w" if p[2] == "weak" else "B")
    # 首锚点向前（段首方向）延伸
    v0 = init_v(anch_seg, ks[0])
    F0 = chain(cands, ks[0], anch_seg[ks[0]], -1, -1, v0, MOT_MIN_FT)
    for j, p in F0.items():
        pos[j] = (p[0], p[1], "w" if p[2] == "weak" else "B")
    # 末锚点向后（段尾方向）延伸
    v1 = init_v(anch_seg, ks[-1])
    F1 = chain(cands, ks[-1], anch_seg[ks[-1]], n, +1, v1, MOT_MIN_FT)
    for j, p in F1.items():
        pos[j] = (p[0], p[1], "w" if p[2] == "weak" else "F")
    return pos, "anch"


def track_free(cands, n):
    """无锚段：取强 ft 候选(score+mot 最优)作种子，双向链；无强种子则整段留空"""
    best_i, best_s, best_c = None, -1, None
    for i in range(n):
        for c in cands[i]:
            if c["tier"] != "strong":
                continue
            s = c["score"] + 0.3 * min(c.get("mot", 0), 3)
            if s > best_s:
                best_s, best_i, best_c = s, i, c
    if best_i is None:
        return {}
    pos = {best_i: (best_c["x"], best_c["y"], "S")}
    F = chain(cands, best_i, (best_c["x"], best_c["y"]), n, +1, [0, 0], MOT_MIN_FT_FREE)
    Bk = chain(cands, best_i, (best_c["x"], best_c["y"]), -1, -1, [0, 0], MOT_MIN_FT_FREE)
    for j, p in F.items():
        pos[j] = (p[0], p[1], "w" if p[2] == "weak" else "F")
    for j, p in Bk.items():
        pos[j] = (p[0], p[1], "w" if p[2] == "weak" else "B")
    return pos


def main():
    only = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None
    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    anch = load_anchors()
    os.makedirs(os.path.join(HERE, "dets_track"), exist_ok=True)

    # ---- 留一法验证：内部锚点隐藏后看链条能否找回来 ----
    loo_hit = loo_tot = 0
    for si, d in anch.items():
        dp_ = os.path.join(HERE, "dets_ft", "%d.json" % si)
        if not os.path.exists(dp_):
            continue
        dets = json.load(open(dp_, encoding="utf-8"))
        cyans = json.load(open(os.path.join(HERE, "dets_cyan", "%d.json" % si), encoding="utf-8"))
        mots = json.load(open(os.path.join(HERE, "dets_mot_ft", "%d.json" % si), encoding="utf-8"))
        cands = build_cands(dets, cyans, mots)
        ks = sorted(d)
        for a in ks:
            prev = [k for k in ks if k < a and a - k >= 4]
            nxt = [k for k in ks if k > a and k - a >= 4]
            if not prev or not nxt:
                continue
            p0, p1 = max(prev), min(nxt)
            d2 = {k: v for k, v in d.items() if k != a}
            sub_anch = {k: v for k, v in d2.items() if p0 <= k <= p1}
            pos, _ = track_seg(sub_anch, cands, len(dets))
            got = pos.get(a)
            loo_tot += 1
            if got and math.hypot(got[0] - d[a][0], got[1] - d[a][1]) < 0.04:
                loo_hit += 1
    if loo_tot:
        print("LOO 验证: %d/%d = %.0f%%" % (loo_hit, loo_tot, 100.0 * loo_hit / loo_tot))

    # ---- 正式跟踪 ----
    for si in range(len(segs)):
        if only is not None and si not in only:
            continue
        dp_ = os.path.join(HERE, "dets_ft", "%d.json" % si)
        if not os.path.exists(dp_):
            continue
        dets = json.load(open(dp_, encoding="utf-8"))
        cyans = json.load(open(os.path.join(HERE, "dets_cyan", "%d.json" % si), encoding="utf-8"))
        mots = json.load(open(os.path.join(HERE, "dets_mot_ft", "%d.json" % si), encoding="utf-8"))
        cands = build_cands(dets, cyans, mots)
        n = len(dets)
        a_seg = anch.get(si, {})
        if a_seg:
            pos, mode = track_seg(a_seg, cands, n)
        else:
            pos = track_free(cands, n)
            mode = "free"
        # 短空洞插值补齐（物理规律近似）：相邻实点间隔 ≤FILL_MAX 用 smoothstep 连接
        ks2 = sorted(pos)
        for a, b in zip(ks2, ks2[1:]):
            gap = b - a - 1
            if 0 < gap <= FILL_MAX:
                p0, p1 = pos[a], pos[b]
                for j in range(a + 1, b):
                    u = (j - a) / (b - a)
                    e = u * u * (3 - 2 * u)
                    pos[j] = (p0[0] + (p1[0] - p0[0]) * e,
                              p0[1] + (p1[1] - p0[1]) * e, "I")
        out = []
        for i, r in enumerate(dets):
            p = pos.get(i)
            if p:
                interp = p[2] == "I"
                sc = {"A": 0.9, "F": 0.6, "B": 0.6, "S": 0.6, "w": 0.4, "I": 0.35}[p[2]]
                out.append({"f": r["f"], "t": r["t"],
                            "ball": {"x": p[0], "y": p[1], "w": 0.008, "h": 0.008,
                                     "score": sc},
                            "real": not interp, "interp": interp, "src": p[2]})
            else:
                out.append({"f": r["f"], "t": r["t"], "ball": None,
                            "real": False, "interp": False})
        with open(os.path.join(HERE, "dets_track", "%d.json" % si), "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False)
        na = sum(1 for r in out if r.get("src") == "A")
        nr = sum(1 for r in out if r["real"]) - na
        nw = sum(1 for r in out if r.get("src") == "w")
        ni = sum(1 for r in out if r.get("interp"))
        print("seg %d [%s] anchors=%d strong=%d weak=%d interp=%d empty=%d covered=%.0f%%" % (
            si, mode, na, nr, nw, ni, n - na - nr - ni,
            100.0 * (na + nr + ni) / n), flush=True)
    print("done")


if __name__ == "__main__":
    main()
