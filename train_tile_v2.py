# -*- coding: utf-8 -*-
"""难例微调：从 ball_tile best.pt 出发，在加入人工锚点难例的 dataset3 上续训。"""
import os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(HERE, "runs", "ball_tile", "weights", "best.pt"))
    model.train(
        data=os.path.join(HERE, "dataset3", "ball.yaml"),
        epochs=50, imgsz=1280, batch=6, device=0,
        project=os.path.join(HERE, "runs"), name="ball_tile_v2", exist_ok=True,
        patience=300, mosaic=0.3, close_mosaic=10, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        lr0=0.0005, lrf=0.05, warmup_epochs=2, cache=False, workers=0,
    )
    print("train tile v2 done")


if __name__ == "__main__":
    main()
