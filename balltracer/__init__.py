# -*- coding: utf-8 -*-
"""BallTracer — 足球(白球)检测与轨迹预计算引擎。

封装自 VCNC 足球跟焦 PC 端预计算管线：
  SAHI 切片检测 + MPEG 式 I/P 帧分区推理 + 物理速度门限时序跟踪
  + 长时轴跳变清理 + 插值补帧。
"""
from ._version import NAME, VERSION, VERSION_DATE, TAGLINE, __version__
from .pipeline import run
from .collect import collect_candidates
from .track import build_track, run_pass
from .annotate import load_cands, draw_video

__all__ = [
    "NAME", "VERSION", "VERSION_DATE", "TAGLINE", "__version__",
    "run", "collect_candidates", "build_track", "run_pass",
    "load_cands", "draw_video",
]
