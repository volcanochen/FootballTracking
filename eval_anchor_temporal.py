# -*- coding: utf-8 -*-
"""长时轴证据融合：三模型融合 + 相邻帧(t±1, t±2)检出并入当前帧候选。
球相邻帧位移 p95<0.02，在匹配半径 R=0.03 内，闪烁检出可被邻帧补回。
用法: eval_anchor_temporal.py [conf=0.05] [tile_weights] [radius=1]
"""
import json, math, os, sys, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.03
NX, OV = 3, 0.20

def full_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = model.predict(up, imgsz=1280, conf=conf, verbose=False)[0]
    return [(float(b.xywhn[0][0]), float(b.xywhn[0][1]), float(b.conf[0]))
            for b in r.boxes if int(b.cls[0]) == 0]

def tile_predict(model, fr, conf):
    up = cv2.resize(fr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    tw = int(UW / (NX - (NX - 1) * OV)); step = tw - int(tw * OV)
    out = []
    for k in range(NX):
        x0 = min(k * step, UW - tw)
        r = model.predict(up[:, x0:x0 + tw], imgsz=1280, conf=conf, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                out.append((float(b.xywh[0][0] + x0) / UW, float(b.xywh[0][1]) / UH, float(b.conf[0])))
    return out

def nms_merge(dets, thr=0.02):
    dets.sort(key=lambda d: -d[2])
    keep = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > thr for k in keep):
            keep.append(d)
    return keep

def main():
    from ultralytics import YOLO
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    tile_w = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'runs', 'ball_tile_v3', 'weights', 'best.pt')
    rad = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    m_full = YOLO(os.path.join(HERE, 'runs', 'ball11n_vb', 'weights', 'best.pt'))
    m_tile = YOLO(tile_w)
    m_old = YOLO(os.path.join(HERE, 'runs', 'ball11n', 'weights', 'best.pt'))
    segs = json.load(open(os.path.join(HERE, 'segs_cur.json'), encoding='utf-8'))
    cap = cv2.VideoCapture(os.path.join(HERE, 'full.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    tot = hit = 0; per = {}
    for si in range(8):
        dp = os.path.join(HERE, 'dets_track/%d.json' % si)
        if not os.path.exists(dp): continue
        d = json.load(open(dp))
        st = [0, 0]
        for a, rec in enumerate(d):
            b = rec.get('ball')
            if not b or not rec.get('real'): continue
            base = int(segs[si][0] * fps) + a * 3
            all_dets = []
            for off in range(-rad, rad + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, base + off)
                ok, fr = cap.read()
                if not ok: continue
                w_decay = 1.0 if off == 0 else 0.6
                all_dets += [(x, y, sc * w_decay) for x, y, sc in
                             nms_merge(full_predict(m_full, fr, conf) + tile_predict(m_tile, fr, conf)
                                       + full_predict(m_old, fr, conf))]
                del fr
            dets = nms_merge(all_dets)
            st[1] += 1
            best = min((math.hypot(x - b['x'], y - b['y']) for x, y, sc in dets), default=9)
            if best < R: st[0] += 1
        per[si] = st; tot += st[1]; hit += st[0]
    cap.release()
    for k in sorted(per):
        print('seg%d: %d/%d (%.0f%%)' % (k, per[k][0], per[k][1], 100.0 * per[k][0] / max(1, per[k][1])))
    print('TEMPORAL(r=%d) conf=%.2f recall: %d/%d (%.1f%%)' % (rad, conf, hit, tot, 100.0 * hit / max(1, tot)))

if __name__ == '__main__':
    main()
