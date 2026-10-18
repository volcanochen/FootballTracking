# -*- coding: utf-8 -*-
"""姿态 pass：yolo11n-pose 提取每帧每人的关键点 → 身体朝向特征。
低清小目标下鼻子/耳朵噪声大，主要用肩线：
  - 肩中点 M、肩向量 S（左肩→右肩），法向 N ⊥ S，符号由鼻子在 M 的哪一侧定
  - facing = N（图像平面内「player 面朝」的水平方向近似）
  - faceConf = |dot(N, nose-M)| / 肩宽：鼻子偏离肩线越远朝向越确定
  - headDown = 鼻子 y 低于肩线 + 0.6 肩宽（低头看脚下球）

输出 dets_pose/<si>.json：[{f, t, ppl: [{x,y,w,h, mx,my, fx,fy, faceConf, headDown}]}]
(x,y,w,h 为关键点外接框，归一化)
"""
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "full.mp4")
segs = json.load(open(os.path.join(HERE, "segs_cur.json"), encoding="utf-8"))
FPS = 29.97
IMGSZ = 640   # 模型固定 640 输入
CONF = 0.15
KPT_CONF = 0.3

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(os.path.join(HERE, "yolo11n-pose.onnx"), so,
                            providers=["CPUExecutionProvider"])
IN_NAME = sess.get_inputs()[0].name


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = im
    return out, r, left, top


def infer_pose(bgr):
    """返回 [{x,y,w,h, mx,my, fx,fy, faceConf, headDown}] 归一化坐标"""
    H0, W0 = bgr.shape[:2]
    lb, r, left, top = letterbox(bgr, IMGSZ)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None]
    out = sess.run(None, {IN_NAME: blob})[0]   # 1x57xN
    pred = out[0].T                             # Nx57
    res = []
    for row in pred:
        sc = row[4]
        if sc < CONF:
            continue
        kpts = row[5:].reshape(17, 3)
        # 像素（letterbox 系）→ 原图归一化
        def pt(i):
            k = kpts[i]
            return (float((k[0] - left) / r / W0), float((k[1] - top) / r / H0),
                    float(k[2]))
        ls, rs = pt(5), pt(6)
        nose = pt(0)
        if ls[2] < KPT_CONF and rs[2] < KPT_CONF:
            continue
        mx, my = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        sw = max(1e-6, ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5)
        # 肩向量法向（两个方向），取指向鼻子一侧
        nx, ny = -(rs[1] - ls[1]), (rs[0] - ls[0])
        nn = ((nx * (nose[0] - mx) + ny * (nose[1] - my)))
        if nn < 0:
            nx, ny = -nx, -ny
        nn = ((nx ** 2 + ny ** 2) ** 0.5) or 1e-6
        fx, fy = nx / nn, ny / nn
        face_conf = min(1.0, abs(nose[0] * 0 + (nx * (nose[0] - mx) + ny * (nose[1] - my))) / sw)
        head_down = nose[1] > my + 0.6 * sw
        xs = [ls[0], rs[0]]
        ys = [ls[1], rs[1]]
        res.append({"x": mx, "y": my,
                    "w": abs(ls[0] - rs[0]) + sw * 0.9, "h": sw * 1.6,
                    "mx": mx, "my": my, "fx": fx, "fy": fy,
                    "faceConf": round(face_conf, 3),
                    "headDown": bool(head_down)})
    return res


def main():
    only = [int(x) for x in os.sys.argv[1].split(",")] if len(os.sys.argv) > 1 else None
    os.makedirs(os.path.join(HERE, "dets_pose"), exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    for si in range(len(segs)):
        if only is not None and si not in only:
            continue
        dp_ = os.path.join(HERE, "dets", "%d.json" % si)
        if not os.path.exists(dp_):
            continue
        dets = json.load(open(dp_, encoding="utf-8"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(segs[si][0] * FPS))
        out = []
        for i in range(len(dets)):
            ok, fr = cap.read()
            if ok:
                for _ in range(2):
                    ok2, _f = cap.read()
                    if not ok2:
                        break
            if not ok:
                out.append({"f": dets[i]["f"], "t": dets[i]["t"], "ppl": []})
                continue
            ppl = infer_pose(fr)
            out.append({"f": dets[i]["f"], "t": dets[i]["t"], "ppl": ppl})
        with open(os.path.join(HERE, "dets_pose", "%d.json" % si), "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False)
        np_avg = sum(len(r["ppl"]) for r in out) / max(1, len(out))
        hd = sum(1 for r in out for p in r["ppl"] if p["headDown"])
        fc = sum(1 for r in out for p in r["ppl"] if p["faceConf"] > 0.3)
        print("seg %d poseAvg=%.1f faceConf>0.3=%d headDown=%d" % (si, np_avg, fc, hd), flush=True)
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
