# -*- coding: utf-8 -*-
"""生成评审图库：每个标注帧 = 原帧全景(854x356, 红框+S/P标签) + 每框 4x 放大插图。
输出 review_gallery/*.jpg + gallery.html（相对路径引用，可直接浏览）。"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LIGHT = (0.630, 0.565)
BAD_SEEDS = {29, 37, 57, 60, 62, 63, 64}

# 1) 重算种子与传播（与 build_dataset2 一致），得到 (seg, i) -> [(x, y, side, src)]
_sv = json.load(open(os.path.join(HERE, "seeds_v2.json"), encoding="utf-8"))
seeds = {}
for k, s in enumerate(_sv):
    if k in BAD_SEEDS:
        continue
    seeds.setdefault((s["seg"], s["i"]), []).append((s["x"], s["y"], 18.0 / 854.0 * 854.0 / 854.0 * 18.0))
# 上面写复杂了：side 单位=854 像素
seeds = {}
for k, s in enumerate(_sv):
    if k in BAD_SEEDS:
        continue
    seeds.setdefault((s["seg"], s["i"]), []).append((s["x"], s["y"], 18.0))

from make_sheets2 import white_candidates  # noqa: E402
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97

white_by_frame = {}
cap = cv2.VideoCapture(os.path.join(HERE, "full.mp4"))
for si in range(8):
    dp_ = os.path.join(HERE, "dets", "%d.json" % si)
    if not os.path.exists(dp_):
        continue
    dets = json.load(open(dp_, encoding="utf-8"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
    fi_video = 0
    idx = 0
    while idx < len(dets):
        ok, fr = cap.read()
        if not ok:
            break
        fi_video += 1
        if fi_video - 1 != idx * 3:
            continue
        white_by_frame[(si, idx)] = fr
        idx += 1

prop = {}
for (si, i0), balls in seeds.items():
    for (x0, y0, s0) in balls:
        for di in range(max(0, i0 - 4), min(len(json.load(open(os.path.join(HERE, "dets", "%d.json" % si)))), i0 + 5)):
            if di == i0:
                continue
            fr = white_by_frame.get((si, di))
            if fr is None:
                continue
            cands = white_candidates(fr)
            best, bd = None, 0.03
            for c in cands:
                d = ((c["x"] - x0) ** 2 + (c["y"] - y0) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, c
            if best:
                prop.setdefault((si, di), []).append((best["x"], best["y"], max(14, 3.5 * best["area"] ** 0.5)))
print("seed frames=%d prop frames=%d" % (len(seeds), len(prop)))

# 2) 逐帧出图
GDIR = os.path.join(HERE, "review_gallery")
os.makedirs(GDIR, exist_ok=True)
for f in os.listdir(GDIR):
    os.remove(os.path.join(GDIR, f))

pages = []
all_keys = sorted(set(list(seeds.keys()) + list(prop.keys())))
for (si, idx) in all_keys:
    fr = white_by_frame.get((si, idx))
    if fr is None:
        continue
    H, W = fr.shape[:2]
    boxes = [(x, y, s, "S") for (x, y, s) in seeds.get((si, idx), [])] + \
            [(x, y, s, "P") for (x, y, s) in prop.get((si, idx), [])]
    vis = fr.copy()
    insets = []
    for bi, (x, y, s854, src) in enumerate(boxes):
        x1, y1 = int((x - s854 / W) * W), int((y - s854 / H) * H)
        x2, y2 = int((x + s854 / W) * W), int((y + s854 / H) * H)
        color = (0, 0, 255) if src == "S" else (255, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, "%s%d" % (src, bi), (x2 + 2, max(10, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        R = 24
        cx, cy = int(x * W), int(y * H)
        t = fr[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
        if t.size:
            t = cv2.resize(t, (R * 8, R * 8), interpolation=cv2.INTER_NEAREST)
            tt = t.copy()
            half = int(s854 * 8 / W / 2)
            cv2.rectangle(tt, (R * 4 - half, R * 4 - int(s854 * 8 / H / 2)),
                          (R * 4 + half, R * 4 + int(s854 * 8 / H / 2)), color, 1)
            cv2.putText(tt, "%s%d" % (src, bi), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            insets.append(tt)
    tag = "seg%d i=%04d" % (si, idx)
    cv2.putText(vis, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    # 拼版：全景在上，插图行在下
    row_h = max((i.shape[0] for i in insets), default=0)
    n_ins = len(insets)
    ins_row = np.zeros((max(row_h, 1), W, 3), np.uint8) if n_ins else None
    if n_ins:
        xo = 4
        for t in insets:
            if xo + t.shape[1] > W:
                break
            ins_row[0:t.shape[0], xo:xo + t.shape[1]] = t
            xo += t.shape[1] + 6
    canvas = np.vstack([vis, ins_row]) if n_ins else vis
    fn = "g_s%d_i%04d.jpg" % (si, idx)
    cv2.imwrite(os.path.join(GDIR, fn), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    pages.append((fn, tag, n_ins))
cap.release()

# 3) 交互式评审 HTML：每帧 [全对]/[全错] 按钮 + 每框 chip 点击循环 未评→对→错
frames_js = [{"id": tag.replace("seg", "s").replace(" i=", "_i"), "tag": tag, "file": fn, "boxes": []} for (fn, tag, n) in pages]
# 重建每帧的框 id 列表（与绘制顺序一致：S 先 P 后）
for (si, idx) in all_keys:
    fid = "s%d_i%04d" % (si, idx)
    for f in frames_js:
        if f["id"] == fid:
            f["boxes"] = ["S%d" % i for i in range(len(seeds.get((si, idx), [])))] + \
                         ["P%d" % i for i in range(len(prop.get((si, idx), [])))]
json.dump(frames_js, open(os.path.join(GDIR, "frames.json"), "w"), ensure_ascii=False)

import json
import base64

for f in frames_js:
    p = os.path.join(GDIR, f["file"])
    f["data"] = "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()

html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>球标注评审</title>",
        "<style>",
        "body{background:#111;color:#ddd;font-family:monospace;margin:0 0 90px 0}",
        "div.f{margin:6px 12px;border-bottom:1px solid #333;padding-bottom:6px}",
        "img{display:block;max-width:1000px;width:100%}",
        "h2{font-size:15px;margin:8px}",
        ".bar{position:fixed;bottom:0;left:0;right:0;background:#222;padding:8px 12px;border-top:2px solid #555;z-index:9}",
        "button{font-size:14px;padding:4px 10px;margin-right:6px;cursor:pointer}",
        ".chip{display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border:1px solid #666;border-radius:10px;cursor:pointer;user-select:none}",
        ".ok{background:#1c5c1c;border-color:#2e2}.bad{background:#6b1c1c;border-color:#e22}",
        "</style></head><body>",
        "<h2>球标注评审（红框 S=种子 / 紫框 P=传播）— 共 %d 帧。操作：点框号循环 未评→对→错；[全对]=整帧通过。[导出结果]后把文本发给 AI</h2>" % len(pages)]
for f in frames_js:
    chips = "".join("<span class='chip' id='c_%s_%s' onclick=\"cyc('%s','%s')\">%s</span>" % (f["id"], b, f["id"], b, b) for b in f["boxes"])
    html.append("<div class='f' id='f_%s'><p>%s <button onclick=\"markAll('%s',1)\">全对</button>"
                "<button onclick=\"markAll('%s',2)\">全错</button> <span id='st_%s'></span></p>"
                "<img src='%s'><div>%s</div></div>" % (f["id"], f["tag"], f["id"], f["id"], f["id"], f["data"], chips))
html.append("""</div>
<div class='bar'>
<span id='prog'></span>
<button onclick='exportResult()'>导出结果</button>
<button onclick='if(confirm("清空全部评审记录?")){localStorage.removeItem("ballrev");init();}'>清空</button>
<textarea id='out' rows=3 style='width:60%;vertical-align:middle'></textarea>
</div>
<script>
var FRAMES = __FRAMES__;
var state = {};
function init(){ try{ state = JSON.parse(localStorage.getItem("ballrev")||"{}"); }catch(e){ state={}; }
  for (var f of FRAMES) for (var b of f.boxes){ var el=document.getElementById("c_"+f.id+"_"+b); if(el){ el.className="chip"+(state[f.id+"/"+b]==1?" ok":state[f.id+"/"+b]==2?" bad":""); } } upd(); }
function cyc(fid,b){ var k=fid+"/"+b; state[k]=(state[k]||0)+1; if(state[k]>2)state[k]=0;
  var el=document.getElementById("c_"+fid+"_"+b); el.className="chip"+(state[k]==1?" ok":state[k]==2?" bad":"");
  localStorage.setItem("ballrev",JSON.stringify(state)); upd(); }
function markAll(fid,v){ for (var f of FRAMES) if (f.id==fid) for (var b of f.boxes){ state[fid+"/"+b]=v;
  var el=document.getElementById("c_"+fid+"_"+b); el.className="chip"+(v==1?" ok":" bad"); }
  localStorage.setItem("ballrev",JSON.stringify(state)); upd(); }
function upd(){ var tot=0,mk=0,bad=0; for(var k in state){ tot++; if(state[k]>0)mk++; if(state[k]==2)bad++; }
  document.getElementById("prog").textContent="框总数:"+tot+" 已评:"+mk+" 错:"+bad; }
function exportResult(){ var bad=[],un=0; for(var f of FRAMES)for(var b of f.boxes){var v=state[f.id+"/"+b]||0; if(v==2)bad.push(f.id+"#"+b); if(v==0)un++;}
  var txt=JSON.stringify({badBoxes:bad,unmarked:un},null,1);
  document.getElementById("out").value=txt; document.getElementById("out").select();
  try{document.execCommand("copy");}catch(e){}
  if(navigator.clipboard) navigator.clipboard.writeText(txt).catch(function(){});
}
init();
</script></body></html>""".replace("__FRAMES__", json.dumps(frames_js, ensure_ascii=False)))
with open(os.path.join(HERE, "review_gallery.html"), "w", encoding="utf-8") as fp:
    fp.write("\n".join(html))
print("pages=%d -> review_gallery.html (interactive)" % len(pages))
