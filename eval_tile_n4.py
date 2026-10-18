# -*- coding: utf-8 -*-
"""测试 NX 切片数对召回的影响。用法: eval_tile_n4.py <weights> <conf> <NX> <OV>"""
import json, math, os, sys, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.03

def main():
    from ultralytics import YOLO
    w, conf = sys.argv[1], float(sys.argv[2])
    NX, OV = int(sys.argv[3]), float(sys.argv[4])
    m = YOLO(w)
    segs = json.load(open(os.path.join(HERE, 'segs_cur.json')))
    cap = cv2.VideoCapture(os.path.join(HERE, 'full.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    tot = hit = 0; per = {}
    for si in range(8):
        dp = os.path.join(HERE, 'dets_track/%d.json' % si)
        if not os.path.exists(dp): continue
        d = json.load(open(dp)); st = [0, 0]
        for a, rec in enumerate(d):
            b = rec.get('ball')
            if not b or not rec.get('real'): continue
            fi = int(segs[si][0] * fps) + a * 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok: continue
            st[1] += 1
            up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            UH, UW = up.shape[:2]; tw = int(UW / (NX - (NX - 1) * OV)); step = tw - int(tw * OV)
            best = 9.0
            for k in range(NX):
                x0 = min(k * step, UW - tw)
                r = m.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
                for bx in r.boxes:
                    if int(bx.cls[0]) != 0: continue
                    x = (float(bx.xywh[0][0]) + x0) / UW; y = float(bx.xywh[0][1]) / UH
                    best = min(best, math.hypot(x - b['x'], y - b['y']))
            if best < R: st[0] += 1
        per[si] = st; tot += st[1]; hit += st[0]
    cap.release()
    print('TILE NX=%d OV=%.2f conf=%.2f recall: %d/%d (%.1f%%)' % (NX, OV, conf, hit, tot, 100.0 * hit / max(1, tot)))
    for k in sorted(per):
        print('seg%d: %d/%d' % (k, per[k][0], per[k][1]))

if __name__ == '__main__':
    main()
