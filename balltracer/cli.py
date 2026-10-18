# -*- coding: utf-8 -*-
"""命令行入口：python -m balltracer ..."""
import argparse

from ._version import NAME, VERSION, TAGLINE
from .pipeline import run
from .config import (CONF, CONF_MIN, INIT_CONF, KEY_EVERY, DEFAULT_TILE_WEIGHTS,
                     DEFAULT_FULL_WEIGHTS, DEFAULT_OLD_WEIGHTS)


def build_parser():
    p = argparse.ArgumentParser(
        prog=NAME,
        description="%s v%s — %s" % (NAME, VERSION, TAGLINE))
    p.add_argument("--video", default="full.mp4", help="源视频路径")
    p.add_argument("--segs", default="segs_cur.json", help="片段清单 JSON（[[start_sec, dur_sec, label], ...]）")
    p.add_argument("--seg-ids", default="0,1,2,3,4,5,6,7", help="逗号分隔的段号，或 'all'")
    p.add_argument("--tile-weights", default=DEFAULT_TILE_WEIGHTS, help="切片检测模型权重")
    p.add_argument("--full-weights", default=DEFAULT_FULL_WEIGHTS, help="全帧 vb 模型权重")
    p.add_argument("--old-weights", default=DEFAULT_OLD_WEIGHTS, help="全帧 old 基线权重")
    p.add_argument("--conf", type=float, default=CONF, help="候选采集置信阈值")
    p.add_argument("--key-every", type=int, default=KEY_EVERY, help="I 帧间隔（P 帧仅 ROI 切片）")
    p.add_argument("--conf-min", type=float, default=CONF_MIN, help="进入门限的最低置信")
    p.add_argument("--init-conf", type=float, default=INIT_CONF, help="起轨最低置信；调低可召回远景弱信号")
    p.add_argument("--out-dir", default=".", help="输出视频目录")
    p.add_argument("--cand-dir", default=None, help="候选缓存目录（默认同 out-dir）")
    p.add_argument("--version", action="version", version="%s %s" % (NAME, VERSION))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    seg_ids = a.seg_ids
    if seg_ids == "all":
        import json
        seg_ids = list(range(len(json.load(open(a.segs, encoding="utf-8")))))
    run(a.video, a.segs, seg_ids, a.tile_weights, a.full_weights, a.old_weights,
        a.conf, a.key_every, a.conf_min, a.init_conf, a.out_dir, a.cand_dir)


if __name__ == "__main__":
    main()
