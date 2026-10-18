# -*- coding: utf-8 -*-
"""变体 A：从旧 best.pt 温和继续训练（更低 lr、弱 mosaic、短程）"""
import os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(HERE, "runs", "ball11n", "weights", "best.pt"))
    model.train(
        data=os.path.join(HERE, "dataset2", "ball_local.yaml"),
        epochs=40, imgsz=1280, batch=6, device=0,
        project=os.path.join(HERE, "runs"), name="ball11n_va", exist_ok=True,
        patience=300, mosaic=0.2, close_mosaic=10, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        lr0=0.0002, lrf=0.1, warmup_epochs=2, cache=False, workers=0,
    )
    print("done A")


if __name__ == "__main__":
    main()
