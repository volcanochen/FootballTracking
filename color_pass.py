# -*- coding: utf-8 -*-
"""颜色检测 pass：独立于 YOLO，对每段每 STEP 帧输出青色球候选 → dets_cyan/<i>.json"""
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(HERE)
import sys  # noqa: E402
sys.path.insert(0, sys_path)
from color_ball import cyan_candidates  # noqa: E402

VIDEO = os.path.join(HERE, "full.mp4")
STEP = 3


def main():
    segs_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "segs_cur.json")
    only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    segs = json.load(open(segs_file, encoding="utf-8"))
    os.makedirs(os.path.join(HERE, "dets_cyan"), exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for si, seg in enumerate(segs):
        if only is not None and si not in only:
            continue
        start, dur = seg[0], seg[1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        out = []
        fi = 0
        n = int(dur * fps)
        while fi < n:
            ok, fr = cap.read()
            if not ok:
                break
            if fi % STEP == 0:
                out.append({"f": fi, "t": round(fi / fps, 3), "cands": cyan_candidates(fr)})
            fi += 1
        with open(os.path.join(HERE, "dets_cyan", "%d.json" % si), "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False)
        avg = sum(len(r["cands"]) for r in out) / max(1, len(out))
        full = sum(1 for r in out if r["cands"])
        print("seg %d frames=%d withCyan=%d avgCands=%.1f" % (si, len(out), full, avg), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
