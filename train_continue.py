# -*- coding: utf-8 -*-
"""继续训练：从上一轮微调权重 runs/ball11n/weights/best.pt 出发，
在 dataset2 上再训练一轮（更低学习率），进一步提升白球检出。
策略与小数据要点沿用 train_ball.py：imgsz=1280、强增广、close_mosaic、小 batch。
"""
import os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    START = os.path.join(HERE, "runs", "ball11n", "weights", "best.pt")
    model = YOLO(START)
    model.train(
        data=os.path.join(HERE, "dataset2", "ball_local.yaml"),
        epochs=80,
        imgsz=1280,
        batch=6,
        device=0,
        project=os.path.join(HERE, "runs"),
        name="ball11n_v2",
        exist_ok=True,
        patience=300,
        mosaic=0.5,
        close_mosaic=20,
        fliplr=0.5,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        lr0=0.0005,
        lrf=0.05,
        warmup_epochs=3,
        cache=False,
        workers=0,
    )
    print("train done")


if __name__ == "__main__":
    main()
