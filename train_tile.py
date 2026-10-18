# -*- coding: utf-8 -*-
"""切片域训练：dataset3（左右 3 切片，2x 上采样）从 COCO 预训练重训。
球在切片内为 9~24px，推理时同域切片，有效分辨率 ~4.5x。"""
import os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(HERE, "weights", "yolo26n.pt"))
    model.train(
        data=os.path.join(HERE, "dataset3", "ball.yaml"),
        epochs=100, imgsz=1280, batch=6, device=0,
        project=os.path.join(HERE, "runs"), name="ball_tile", exist_ok=True,
        patience=300, mosaic=0.5, close_mosaic=20, fliplr=0.5,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        lr0=0.002, lrf=0.01, warmup_epochs=5, cache=False, workers=0,
    )
    print("train tile done")


if __name__ == "__main__":
    main()
