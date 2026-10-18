# -*- coding: utf-8 -*-
"""PC 预计算流水线 · 检测层（python）
输入：full.mp4 + segs.json（段列表 [start, dur, note]）
输出：dets/<i>.json（每 3 帧的 YOLO 检测）+ frames/<i>_<f>.rgba（128×72，node 侧跑 detectGoal 用）

检测：yolo11n.onnx（COCO）+ 2x 上采样到 1280 letterbox。
仅个人学习验证，视频/权重不外传。
"""
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "yolo_baseline"))
from baseline import infer_upscale  # noqa

VIDEO = os.path.join(HERE, "full.mp4")
STEP = 3  # 每 3 帧推理一次（10fps，相机缓动足够）


def main():
    segs_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "segs.json")
    only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    segs = json.load(open(segs_file, encoding="utf-8"))
    os.makedirs(os.path.join(HERE, "dets"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "frames"), exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print("video fps=%.3f" % fps)

    for si, seg in enumerate(segs):
        if only is not None and si not in only:
            continue
        start, dur = seg[0], seg[1]
        f0 = int(start * fps)
        n = int(dur * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        dets = []
        fi = 0
        while fi < n:
            ok, fr = cap.read()
            if not ok:
                break
            if fi % STEP == 0:
                # imgsz 1920 + conf 0.18：低清小目标多换检出，误检交由 tracks.js 时序层过滤
                d = infer_upscale(fr, 2.0, size=1920, conf=0.18)
                ball = [x for x in d if x["cls"] == "sports ball"]
                people = [x for x in d if x["cls"] == "person" and x["w"] > 0.015 and x["h"] > 0.03]
                rec = {
                    "f": fi, "t": round(fi / fps, 3),
                    "ball": max(ball, key=lambda x: x["score"]) if ball else None,
                    "ballAll": sorted(ball, key=lambda x: -x["score"])[:3],
                    "people": people,
                }
                dets.append(rec)
                small = cv2.resize(fr, (128, 72), interpolation=cv2.INTER_AREA)
                with open(os.path.join(HERE, "frames", "%d_%04d.rgba" % (si, fi)), "wb") as fp:
                    fp.write(cv2.cvtColor(small, cv2.COLOR_BGR2RGBA).tobytes())
            fi += 1
        with open(os.path.join(HERE, "dets", "%d.json" % si), "w", encoding="utf-8") as fp:
            json.dump(dets, fp, ensure_ascii=False)
        nb = sum(1 for r in dets if r["ball"])
        print("seg %d [%ds,%ds] frames=%d ballFrames=%d peopleAvg=%.1f" %
              (si, start, start + dur, len(dets), nb,
               sum(len(r["people"]) for r in dets) / max(1, len(dets))), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
