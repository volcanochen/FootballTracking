# -*- coding: utf-8 -*-
"""负样本强化微调：从 ball_tile_v2 出发，加入 630 张定向负样本切片（禁区线等静态误检）。"""
import os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(HERE, "runs", "ball_tile_v2", "weights", "best.pt"))
    model.train(
        data=os.path.join(HERE, "dataset3", "ball.yaml"),
        epochs=40, imgsz=1280, batch=6, device=0,
        project=os.path.join(HERE, "runs"), name="ball_tile_v3", exist_ok=True,
        patience=300, mosaic=0.3, close_mosaic=10, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        lr0=0.0004, lrf=0.05, warmup_epochs=2, cache=False, workers=0,
    )
    print("train tile v3 done")


if __name__ == "__main__":
    main()
