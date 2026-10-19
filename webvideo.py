#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器可播的 MP4 写入器（H.264 + yuv420p + faststart）。

cv2.VideoWriter 在 Windows 上通常只有 mp4v(MPEG-4 Part 2) 可用，
avc1 需要额外的 openh264 DLL，缺失时会静默退回 mp4v —— 产出的文件
浏览器 / 飞书 / 网盘预览都不认，只能下载后本地播放。

本模块用 PyAV 内置的 libx264 直接编码，接口与 cv2.VideoWriter 兼容：

    from webvideo import WebWriter
    vw = WebWriter("out.mp4", fps, (1280, 720))
    vw.write(bgr_frame)      # numpy BGR，和 cv2 一致
    vw.release()

依赖：pip install av （PyAV 已自带 libx264，无需单独装 ffmpeg）
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

try:
    import av
    _AV_ERR = None
except Exception as _e:  # noqa: BLE001  PyAV 缺失时退化为 cv2
    av = None
    _AV_ERR = _e


class WebWriter:
    """替代 cv2.VideoWriter，输出浏览器可直接 <video> 播放的 H.264 MP4。"""

    def __init__(self, path: str, fps: float, size: tuple[int, int], crf: int = 23,
                 preset: str = "medium"):
        w, h = int(size[0]), int(size[1])
        if av is None:
            raise RuntimeError(f"PyAV 未安装: {_AV_ERR}")
        if w % 2 or h % 2:
            raise ValueError(f"尺寸必须为偶数（H.264 yuv420p 限制）: {w}x{h}")
        self.path = path
        self.size = (w, h)
        fps = Fraction(fps).limit_denominator(1001) or Fraction(25)
        self.fps = fps
        self._n = 0

        self._out = av.open(path, "w", format="mp4", options={"movflags": "+faststart"})
        self._st = self._out.add_stream("libx264", rate=fps)
        self._st.width, self._st.height = w, h
        self._st.pix_fmt = "yuv420p"
        self._st.options = {
            "crf": str(crf),
            "preset": preset,
            "profile": "high",
            "x264-params": "keyint=60:min-keyint=15:scenecut=40:ref=3",
        }
        self._st.codec_context.time_base = Fraction(1, 1) / fps
        self._stopped = False

    def isOpened(self) -> bool:  # noqa: N802 - 对齐 cv2 接口
        return self._out is not None

    def write(self, frame: np.ndarray) -> None:
        if self._stopped:
            raise RuntimeError("WebWriter 已关闭")
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"帧尺寸 {w}x{h} 与初始化的 {self.size} 不一致")
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        vf = vf.reformat(format="yuv420p")
        vf.pts = self._n
        vf.time_base = Fraction(1, 1) / self.fps
        for pkt in self._st.encode(vf):
            self._out.mux(pkt)
        self._n += 1

    def release(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for pkt in self._st.encode(None):
            self._out.mux(pkt)
        self._out.close()

    # 上下文管理器写法
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def open_writer(path: str, fps: float, size: tuple[int, int], crf: int = 23):
    """有 PyAV 就用 H.264；没有则退回 cv2（并提示需要转码）。"""
    try:
        return WebWriter(path, fps, size, crf=crf)
    except Exception as e:  # noqa: BLE001
        import cv2

        print(f"[webvideo] PyAV 不可用({e})，退回 cv2 mp4v —— 该文件浏览器无法在线播放")
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


if __name__ == "__main__":
    # 自检：写一个 1 秒的渐变测试片
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "_webvideo_selftest.mp4"
    with WebWriter(out, 25, (320, 240)) as wr:
        for i in range(25):
            wr.write(np.full((240, 320, 3), i * 8, np.uint8))
    print("wrote", out)
