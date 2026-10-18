# -*- coding: utf-8 -*-
"""模型单帧检测标注视频：不加跟踪/插值，纯看模型逐帧能否检出球。
用法: python anno_model_video.py <weights> [seg_ids=2,3] [conf=0.10]
红框 = 该帧最高分球检出（旁标 conf）；橙细框 = 其余候选。
输出: model_anno_s<i>.mp4
"""
import json
import os
import sys

import cv2


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    weights = sys.argv[1]
    seg_ids = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,3").split(",")]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
    VIDEO = os.path.join(HERE, "full.mp4")
    model = YOLO(weights)
    tag = os.path.basename(os.path.dirname(os.path.dirname(weights)))  # runs/<tag>/weights/best.pt
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    for si in seg_ids:
        start, dur = segs[si][0], segs[si][1]
        n_frames = int(dur * fps)
        out_path = os.path.join(HERE, "model_anno_s%d_%s.mp4" % (si, tag))
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (854, 356))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        n_hit = n_hi = 0
        for vf in range(n_frames):
            ok, fr = cap.read()
            if not ok:
                break
            up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
            balls = []
            for b in r.boxes:
                if int(b.cls[0]) != 0:
                    continue
                x, y, w, h = [float(v) for v in b.xywhn[0]]  # 归一化坐标与缩放无关
                balls.append((x, y, w, h, float(b.conf[0])))
            balls.sort(key=lambda d: -d[4])
            if balls:
                n_hit += 1
                if balls[0][4] > 0.5:
                    n_hi += 1
            H, W = fr.shape[:2]
            for k, (x, y, w, h, sc) in enumerate(balls):
                px, py = int((x - w / 2) * W), int((y - h / 2) * H)
                pw, ph = max(4, int(w * W)), max(4, int(h * H))
                if k == 0:
                    cv2.rectangle(fr, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
                    cv2.putText(fr, "ball %.2f" % sc, (px, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                else:
                    cv2.rectangle(fr, (px, py), (px + pw, py + ph), (0, 160, 255), 1)
            cv2.putText(fr, "seg%d %.1fs" % (si, vf / fps), (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            wr.write(fr)
        wr.release()
        print("saved %s  ballFrames=%d/%d (%.0f%%)  hiConf(>0.5)=%d" %
              (out_path, n_hit, n_frames, 100.0 * n_hit / max(1, n_frames), n_hi), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
