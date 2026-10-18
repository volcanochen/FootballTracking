# -*- coding: utf-8 -*-
"""微调训练：yolo11n（COCO 预训练）→ 本场地白球领域权重。
小数据策略：imgsz=1280（球在 854 源里仅 3~8px，2x 放大后仍小）、
强增广、close_mosaic 关闭以免小目标被 mosaic 稀释、冻结 backbone 不需要（数据少全微调）。
"""
import os
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(HERE, "yolo11n.pt"))
model.train(
    data=os.path.join(HERE, "dataset2", "ball.yaml"),
    epochs=60,
    imgsz=1280,
    batch=6,
    device=0,
    project=os.path.join(HERE, "runs"),
    name="ball11n",
    exist_ok=True,
    patience=200,
    mosaic=0.5,
    close_mosaic=20,
    fliplr=0.5,
    hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
    lr0=0.002,
    cache=False,
    workers=0,
)
print("train done")
