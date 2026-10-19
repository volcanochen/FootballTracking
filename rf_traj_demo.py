#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RF-DETR 高置信度分档 + 全程持久轨迹演示（评审用）.

流程（两遍式，推理只跑一次）:
  Pass1  每段以 low_thr=0.05 全量推理, 缓存每帧最佳球 (conf, cx, cy) -> cache_s{si}.json
  Pass2  对每个置信度档位 C（默认 45/50/55/60/65%）:
     - 检出帧:   最佳球 conf >= C          -> 轨迹点 kind='det'    (绿)
     - 缺检帧:   前后最近两个检出帧线性插值 -> kind='interp'       (橙)
     - 段首之前/段尾之后(到头了): 保持最近检出位置(随便猜一个) -> kind='guess' (品红)
     - 绘制: 全程轨迹线(不随时间消失) + 每 12 帧一个方向箭头 + 当前帧球框(仅检出帧)
     - 编码 rftraj_s{si}_c{C}.mp4, 并抽取各段最高置信帧静帧
  最后生成评审页 review.html（相对路径引用视频与静帧）。

用法:
  python rf_traj_demo.py --seg-ids 0,2,3,5,6 --levels 0.45,0.50,0.55,0.60,0.65 --out-dir rf_out45
"""
import os
import sys
import json
import math
import argparse
import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from webvideo import open_writer  # noqa: E402  浏览器可播的 H.264 写入器

ARROW_EVERY = 12          # 每隔多少帧画一个方向箭头
LEVELS_DEFAULT = "0.45,0.50,0.55,0.60,0.65"

# BGR
COL = {
    "det":    (0, 220, 0),     # 绿: 实检
    "interp": (0, 165, 255),   # 橙: 前后检出帧线性插值
    "guess":  (255, 0, 255),   # 品红: 到头了保持最近检出位置(猜测)
}
KIND_CN = {"det": "实检", "interp": "插值", "guess": "猜测"}
BALL_COLOR = (0, 0, 255)


# ---------------------------------------------------------------------------
# 轨迹构建
# ---------------------------------------------------------------------------
def build_traj(best, level):
    """best: 每帧 [conf, cx, cy] 或 None; 返回 (traj, stats).

    traj: 每帧 (x, y, kind) 或 None（本段无任何检出时全 None）。
    """
    n = len(best)
    det = [i for i, b in enumerate(best) if b and b[0] >= level]
    traj = [None] * n
    for i in det:
        cx, cy = best[i][1], best[i][2]
        traj[i] = (cx, cy, "det")
    # 缺检帧: 前后最近检出帧线性插值
    max_gap = 0
    for a, b in zip(det[:-1], det[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        max_gap = max(max_gap, gap)
        ax, ay = best[a][1], best[a][2]
        bx, by = best[b][1], best[b][2]
        for k in range(1, gap + 1):
            t = k / float(b - a)
            traj[a + k] = (ax + (bx - ax) * t, ay + (by - ay) * t, "interp")
    # 到头了: 保持最近检出位置(猜测)
    n_head = n_tail = 0
    if det:
        fx, fy = best[det[0]][1], best[det[0]][2]
        for i in range(det[0]):
            traj[i] = (fx, fy, "guess")
            n_head += 1
        lx, ly = best[det[-1]][1], best[det[-1]][2]
        for i in range(det[-1] + 1, n):
            traj[i] = (lx, ly, "guess")
            n_tail += 1
    stats = {
        "frames": n,
        "det": len(det),
        "interp": max(0, n - len(det) - n_head - n_tail),
        "guess": n_head + n_tail,
        "guess_head": n_head,
        "guess_tail": n_tail,
        "max_gap": max_gap,
        "det_pct": round(100.0 * len(det) / n, 1) if n else 0.0,
        "avg_conf": round(float(np.mean([best[i][0] for i in det])), 3) if det else 0.0,
        "max_conf": round(max((best[i][0] for i in det), default=0.0), 3),
        "first_det": det[0] if det else None,
        "last_det": det[-1] if det else None,
    }
    return traj, stats


def seg_kind(kind_a, kind_b):
    """两点连线取色: 只要有一端不是 det, 就用非 det 的颜色。"""
    return kind_a if kind_a != "det" else kind_b


def draw_frame(frame, traj, upto, box=None, conf=None, hud=""):
    """画 [0, upto] 的全程轨迹 + 箭头 + 当前帧球框 + HUD."""
    H, W = frame.shape[:2]
    # 1) 轨迹线（不消失）
    prev = None
    for i in range(upto + 1):
        p = traj[i]
        if p is None:
            prev = None
            continue
        x, y, kind = p
        pt = (int(round(x)), int(round(y)))
        if prev is not None:
            cv2.line(frame, prev[0], pt, COL[seg_kind(prev[1], kind)], 1, cv2.LINE_AA)
        cv2.circle(frame, pt, 1, COL[kind], -1, cv2.LINE_AA)
        prev = (pt, kind)
    # 2) 方向箭头（每 ARROW_EVERY 帧一个）
    for i in range(2, upto + 1):
        if i % ARROW_EVERY:
            continue
        p0, p1 = traj[i - 2], traj[i]
        if p0 is None or p1 is None:
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        nrm = math.hypot(dx, dy)
        if nrm < 1e-3:
            continue
        c = COL[p1[2]]
        base = (int(round(p1[0])), int(round(p1[1])))
        tip = (int(round(p1[0] + dx / nrm * 14)), int(round(p1[1] + dy / nrm * 14)))
        cv2.arrowedLine(frame, base, tip, c, 2, cv2.LINE_AA, tipLength=0.45)
    # 3) 当前帧球框（仅检出帧）
    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 3)
        lab = f"ball {conf:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + len(lab) * 9 + 6, y1), BALL_COLOR, -1)
        cv2.putText(frame, lab, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    # 4) HUD
    cv2.rectangle(frame, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(frame, hud, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    # 5) 图例（右上，单行）
    legend = [("det", "DET"), ("interp", "INTP"), ("guess", "GUESS")]
    x = W - 230
    for kind, tag in legend:
        cv2.circle(frame, (x, 13), 4, COL[kind], -1)
        cv2.putText(frame, tag, (x + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL[kind], 1, cv2.LINE_AA)
        x += 72
    return frame


# ---------------------------------------------------------------------------
# Pass1: 推理缓存
# ---------------------------------------------------------------------------
def build_cache(seg_ids, segs, weights, device, low_thr):
    from rf_soccernet_demo import build_model, detect_frame
    model = build_model(weights, device)
    caches = {}
    for sid in seg_ids:
        cp = os.path.join(OUT_DIR, f"cache_s{sid}.json")
        if os.path.exists(cp):
            caches[sid] = json.load(open(cp, encoding="utf-8"))
            print(f"[cache] seg{sid} 命中缓存 ({caches[sid]['n']} 帧)")
            continue
        seg = segs[sid]
        cap = cv2.VideoCapture(VIDEO)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        s0 = int(seg[0] * vfps)
        s1 = int((seg[0] + seg[1]) * vfps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        best = []
        fi = s0
        while fi < s1:
            ok, fr = cap.read()
            if not ok:
                break
            dets = detect_frame(model, fr, device, low_thr)
            bb = None
            for d in dets:
                if d[0] == "ball" and (bb is None or d[5] > bb[0]):
                    cx, cy = (d[1] + d[3]) / 2.0, (d[2] + d[4]) / 2.0
                    bb = [round(d[5], 4), round(cx, 1), round(cy, 1),
                          round(d[1], 1), round(d[2], 1), round(d[3], 1), round(d[4], 1)]
            best.append(bb)
            fi += 1
        cap.release()
        caches[sid] = {"vfps": vfps, "s0": s0, "W": W, "H": H, "n": len(best), "best": best}
        json.dump(caches[sid], open(cp, "w", encoding="utf-8"))
        nd = sum(1 for b in best if b)
        print(f"[cache] seg{sid} 推理完成 {len(best)} 帧, 有球帧 {nd}")
    return caches


# ---------------------------------------------------------------------------
# Pass2: 分档绘制
# ---------------------------------------------------------------------------
def render_level(caches, seg_ids, segs, level, out_dir):
    lv = int(round(level * 100))
    all_stats = {}
    for sid in seg_ids:
        c = caches[sid]
        best = c["best"]
        traj, st = build_traj(best, level)
        all_stats[sid] = st
        if st["det"] == 0:
            print(f"  [c{lv} seg{sid}] 无检出, 跳过视频")
            continue
        out_path = os.path.join(out_dir, f"rftraj_s{sid}_c{lv}.mp4")
        # 用 webvideo 输出 H.264/faststart：mp4v 浏览器无法在线预览
        vw = open_writer(out_path, c["vfps"], (c["W"], c["H"]))
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["s0"])
        still_frame_idx = None
        if st["det"]:
            still_frame_idx = max((i for i in range(len(best)) if best[i] and best[i][0] >= level),
                                  key=lambda i: best[i][0])
        still_saved = False
        fi_local = 0
        counts = {"det": 0, "interp": 0, "guess": 0}
        while True:
            ok, fr = cap.read()
            if not ok or fi_local >= len(traj):
                break
            b = best[fi_local]
            box = conf = None
            if b and b[0] >= level:
                box = (b[3], b[4], b[5], b[6])
                conf = b[0]
            hud = (f"RF-DETR | seg{sid} {segs[sid][2] if len(segs[sid]) > 2 else ''} | "
                   f"conf>={level:.2f} | det {counts['det']} intp {counts['interp']} "
                   f"guess {counts['guess']}")
            draw_frame(fr, traj, fi_local, box, conf, hud)
            vw.write(fr)
            # 抽最高置信帧静帧
            if not still_saved and still_frame_idx is not None and fi_local == still_frame_idx:
                cv2.imwrite(os.path.join(out_dir, "frames", f"s{sid}_c{lv}_best.png"), fr)
                still_saved = True
            p = traj[fi_local]
            if p is not None:
                counts[p[2]] += 1
            fi_local += 1
        cap.release()
        vw.release()
        print(f"  [c{lv} seg{sid}] {out_path}  det={st['det']} intp={st['interp']} "
              f"guess={st['guess']} max_gap={st['max_gap']}")
    return all_stats


# ---------------------------------------------------------------------------
# HTML 评审页
# ---------------------------------------------------------------------------
def build_html(out_dir, seg_ids, segs, levels, stats_by_level):
    lv_tags = [int(round(l * 100)) for l in levels]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # 矩阵表
    mat_rows = ""
    for sid in seg_ids:
        cells = ""
        for l, lv in zip(levels, lv_tags):
            st = stats_by_level[l].get(sid)
            if st is None or st["det"] == 0:
                cells += '<td class="muted">无检出</td>'
                continue
            cls = "good" if st["det_pct"] >= 55 else ("mid" if st["det_pct"] >= 35 else "bad")
            cells += (f'<td class="{cls}"><b>{st["det_pct"]}%</b><br>'
                      f'<small>实检 {st["det"]} · 插值 {st["interp"]} · 猜测 {st["guess"]}'
                      f' · 最大空档 {st["max_gap"]} 帧</small></td>')
        label = segs[sid][2] if len(segs[sid]) > 2 else ""
        mat_rows += f"<tr><td>seg{sid} <small>{label}</small></td>{cells}</tr>"

    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RF-DETR 高置信度分档 · 全程轨迹评审</title>
<style>
  body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 24px auto;
         max-width: 1180px; color: #1a1a2e; background: #f7f8fa; line-height: 1.55; }}
  h1 {{ font-size: 22px; border-bottom: 3px solid #c0392b; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; margin-top: 36px; color: #c0392b; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
  th, td {{ border: 1px solid #d8dbe2; padding: 6px 9px; text-align: center; }}
  th {{ background: #eef1f5; }}
  td.good {{ background: #e8f6e8; }} td.mid {{ background: #fff7e0; }} td.bad {{ background: #fdeaea; }}
  td.muted {{ color: #999; }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  .legend span {{ display: inline-block; padding: 2px 12px; border-radius: 10px;
                  color: #fff; margin-right: 10px; font-size: 13px; }}
  .lg-det {{ background: #00b400; }} .lg-int {{ background: #ffa500; }} .lg-gue {{ background: #d400d4; }}
  .vids {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .card {{ background: #fff; border: 1px solid #d8dbe2; border-radius: 8px; padding: 10px; }}
  .card h3 {{ margin: 2px 0 8px; font-size: 14px; }}
  video {{ width: 100%; border-radius: 4px; background: #000; }}
  .stills img {{ width: 30%; border: 1px solid #ccc; border-radius: 4px; margin: 4px; }}
  .note {{ background: #fff8e6; border-left: 4px solid #f0a500; padding: 10px 14px;
           font-size: 13.5px; margin: 12px 0; }}
  .meta {{ color: #777; font-size: 12.5px; }}
</style>
</head>
<body>
<h1>RF-DETR-SoccerNet 高置信度分档 · 全程球轨迹评审</h1>
<p class="meta">生成时间 {now} · 视频 854×356 夜场业余源 · 推理 low_thr=0.05 全量缓存,
显示/统计按档位过滤 · 脚本 rf_traj_demo.py</p>

<div class="note"><b>方法与图例</b>：每帧取置信度最高的球候选；≥档位阈值记 <span class="legend">
<span class="lg-det">实检(绿)</span></span>，缺检帧用前后最近两个实检帧<b>线性插值</b>补上
<span class="legend"><span class="lg-int">插值(橙)</span></span>，段首之前/段尾之后找不到检出帧时
<b>保持最近检出位置</b>（即"到头了随便猜一个"）<span class="legend">
<span class="lg-gue">猜测(品红)</span></span>。轨迹全程保留不消失，每 12 帧画一个方向箭头，
实检帧同时画红色球框。注意：轨迹未做任何物理门限/平滑，跳变会如实保留——这正是要评审的内容。</div>

<h2>1. 分档 × 片段 覆盖率矩阵（每帧最佳球 ≥ 档位阈值的帧占比）</h2>
<table>
<tr><th>片段 \\ 档位</th><th>≥45%</th><th>≥50%</th><th>≥55%</th><th>≥60%</th><th>≥65%</th></tr>
{mat_rows}
</table>
<p class="meta">单元格内：实检/插值/猜测帧数与最大连续缺检空档（帧）。29.97fps 下 30 帧空档 ≈ 1 秒无实检。</p>
"""
    # 分档详情
    for l, lv in zip(levels, lv_tags):
        sts = stats_by_level[l]
        tot_det = sum(sts[s]["det"] for s in seg_ids if s in sts)
        tot_frm = sum(sts[s]["frames"] for s in seg_ids if s in sts)
        body += f'<h2>2.{lv_tags.index(lv) + 1} 档位 conf ≥ {lv}%（五段合计实检覆盖 {100.0 * tot_det / tot_frm:.1f}%）</h2>\n'
        body += '<div class="vids">'
        for sid in seg_ids:
            st = sts.get(sid)
            if st is None or st["det"] == 0:
                continue
            label = segs[sid][2] if len(segs[sid]) > 2 else ""
            body += f"""<div class="card">
<h3>seg{sid} · {label} · 实检 {st["det_pct"]}%（{st["det"]}/{st["frames"]} 帧）
· 插值 {st["interp"]} · 猜测 {st["guess"]} · 平均 conf {st["avg_conf"]}</h3>
<video controls preload="metadata" src="rftraj_s{sid}_c{lv}.mp4"></video>
</div>"""
        body += "</div>"
        # 静帧
        body += '<div class="stills"><p><b>各段最高置信帧截图：</b></p>'
        for sid in seg_ids:
            p = f"frames/s{sid}_c{lv}_best.png"
            if os.path.exists(os.path.join(out_dir, p)):
                st = sts[sid]
                body += (f'<figure style="display:inline-block;margin:4px;text-align:center">'
                         f'<img src="{p}" alt="seg{sid}"><br>'
                         f'<small>seg{sid} max conf {st["max_conf"]}</small></figure>')
        body += "</div>"

    body += """
<h2>3. 评审关注点</h2>
<ol>
<li><b>档位-覆盖权衡</b>：档位每提高 5%，实检覆盖掉多少？哪个档位开始"猜测段"占比失控？</li>
<li><b>轨迹跳变</b>：橙色/绿色轨迹里出现的长距离直线跳变 = 高置信误检（0.40 档精度仅 7.6% 的余毒）。
数一数每段几次明显跳变，判断是否能用物理门限（速度上限）清掉。</li>
<li><b>品红段长度</b>：段首/段尾的品红段是"硬猜"，长度 = 开头/结尾连续无检出的时间。
若品红段过长（&gt;2s），说明该档位在该段基本不可用。</li>
<li><b>插值合理性</b>：橙段的直线连接是否符合球的实际运动（近距离短空档通常成立；
长空档 + 球变向时会画出不符合实际的直线）。</li>
<li><b>与 BallTracer 对照</b>：BallTracer 单帧召回 76.2%（R=0.02）。看哪个档位的实检覆盖
+ 物理门限后最接近，以及 RF-DETR 候选是否能当 BallTracer 的第二来源。</li>
</ol>
<p class="meta">RF-DETR-SoccerNet（HF julianzu9612/RFDETR-Soccernet）· 架构重建与基线对比见
docs/EVAL_RFDETR_VS_BALLTRACER.md</p>
</body></html>"""
    html_path = os.path.join(out_dir, "review.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"=== 评审页已生成: {html_path} ===")
    return html_path


# ---------------------------------------------------------------------------
def main():
    global OUT_DIR, VIDEO
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rfdetr_soccernet/weights/checkpoint_best_regular.pth")
    ap.add_argument("--video", default="full.mp4")
    ap.add_argument("--segs", default="segs_cur.json")
    ap.add_argument("--seg-ids", default="0,2,3,5,6")
    ap.add_argument("--levels", default=LEVELS_DEFAULT)
    ap.add_argument("--low-thr", type=float, default=0.05)
    ap.add_argument("--out-dir", default="rf_out45")
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    VIDEO = args.video
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    segs = json.load(open(args.segs, encoding="utf-8"))
    seg_ids = [int(x) for x in args.seg_ids.split(",")]
    levels = [float(x) for x in args.levels.split(",")]

    caches = build_cache(seg_ids, segs, args.weights, device, args.low_thr)

    stats_by_level = {}
    for level in levels:
        lv = int(round(level * 100))
        print(f"=== 渲染档位 conf>={level:.2f} ===")
        stats_by_level[level] = render_level(caches, seg_ids, segs, level, OUT_DIR)
        json.dump(stats_by_level[level], open(os.path.join(OUT_DIR, f"traj_stats_c{lv}.json"),
                                              "w", encoding="utf-8"), indent=2)

    build_html(OUT_DIR, seg_ids, segs, levels, stats_by_level)
    print("=== 全部完成 ===")


if __name__ == "__main__":
    main()
