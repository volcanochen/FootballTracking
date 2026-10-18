# -*- coding: utf-8 -*-
"""基线评测：指定权重在 dataset2/val 上的 mAP。用法: python eval_val.py <weights_path>"""
import sys, os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    w = sys.argv[1]
    model = YOLO(w)
    m = model.val(
        data=os.path.join(HERE, "dataset2", "ball_local.yaml"),
        imgsz=1280, device=0, batch=4, plots=False, verbose=True,
    )
    print("=== %s ===" % os.path.basename(os.path.dirname(os.path.dirname(w))))
    print("mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f" % (m.box.map50, m.box.map, m.box.mp, m.box.mr))


if __name__ == "__main__":
    main()
