# -*- coding: utf-8 -*-
"""变体 B：从 COCO 预训练重新微调（更长 epochs，其余沿用原配方）"""
import os


def main():
    from ultralytics import YOLO
    HERE = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(HERE, "weights", "yolo26n.pt"))
    model.train(
        data=os.path.join(HERE, "dataset2", "ball_local.yaml"),
        epochs=150, imgsz=1280, batch=6, device=0,
        project=os.path.join(HERE, "runs"), name="ball11n_vb", exist_ok=True,
        patience=300, mosaic=0.5, close_mosaic=30, fliplr=0.5,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        lr0=0.002, lrf=0.01, warmup_epochs=5, cache=False, workers=0,
    )
    print("done B")


if __name__ == "__main__":
    main()
