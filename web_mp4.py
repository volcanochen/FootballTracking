#!/usr/bin/env python
"""把 OpenCV 写的 mp4v 视频转成浏览器可直接在线播放的 H.264 / MP4。

背景：cv2.VideoWriter(..., fourcc("mp4v")) 产出的是 MPEG-4 Part 2，
Chrome / Edge / Firefox / Safari 都不支持解码，页面里只报
「该视频编码格式浏览器不支持在线预览」；另外 moov 索引默认在文件尾，
不加 faststart 也无法边下边播。

本脚本只依赖 PyAV（已内置 libx264），不需要单独装 ffmpeg。

用法：
    python web_mp4.py rf_out45                 # 整个目录，原地替换，原件备份到 .mp4v_backup/
    python web_mp4.py rf_out45 --dry-run       # 只看哪些需要转，不动文件
    python web_mp4.py rf_out45 --crf 20        # 更高质量（默认 23）
    python web_mp4.py rf_out45 --preset slow   # 更小体积（默认 medium）
    python web_mp4.py a.mp4 b.mp4 --outdir web # 输出到指定目录，不覆盖原件
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import struct
import sys
import time
from fractions import Fraction

import av


# --------------------------------------------------------------------------
# MP4 容器探测：读 stsd 里的编码 fourcc + moov 是否在头部
# --------------------------------------------------------------------------
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}


def probe(path: str) -> dict:
    info = {"codec": None, "faststart": None, "size": os.path.getsize(path)}
    with open(path, "rb") as f:
        data = f.read()

    def walk(off: int, end: int) -> None:
        while off < end - 8:
            try:
                size, typ = struct.unpack(">I4s", data[off:off + 8])
            except struct.error:
                return
            if size < 8:
                return
            if typ in _CONTAINERS:
                walk(off + 8, off + size)
            elif typ == b"stsd" and info["codec"] is None:
                try:
                    cnt = struct.unpack(">I", data[off + 12:off + 16])[0]
                    p = off + 16
                    for _ in range(cnt):
                        s = struct.unpack(">I", data[p:p + 4])[0]
                        info["codec"] = data[p + 4:p + 8].decode("latin1")
                        p += s
                except struct.error:
                    pass
            off += size

    walk(0, len(data))
    info["faststart"] = data.find(b"moov") < data.find(b"mdat")
    return info


def is_web_ready(path: str) -> bool:
    i = probe(path)
    return i["codec"] in ("avc1", "avc3", "hev1", "hvc1") and i["faststart"] is True


# --------------------------------------------------------------------------
# 转码
# --------------------------------------------------------------------------
def transcode(src: str, dst: str, crf: int = 23, preset: str = "medium") -> dict:
    inp = av.open(src)
    ist = inp.streams.video[0]
    ist.thread_type = "AUTO"

    fps = ist.average_rate or ist.guessed_rate or Fraction(25)
    w, h = ist.codec_context.width, ist.codec_context.height
    if w % 2 or h % 2:
        raise ValueError(f"{src}: 尺寸 {w}x{h} 不是偶数，H.264 yuv420p 不支持")
    frames = ist.frames or 0

    out = av.open(dst, "w", format="mp4", options={"movflags": "+faststart"})
    ost = out.add_stream("libx264", rate=fps)
    ost.width, ost.height = w, h
    ost.pix_fmt = "yuv420p"
    # profile high + yuv420p = 浏览器/移动端最通用的组合
    ost.options = {
        "crf": str(crf),
        "preset": preset,
        "profile": "high",
        "x264-params": "keyint=60:min-keyint=15:scenecut=40:ref=3",
    }
    ost.codec_context.time_base = Fraction(1, 1) / fps

    n = 0
    t0 = time.time()
    for frame in inp.decode(ist):
        frame = frame.reformat(format="yuv420p")
        frame.pts = n
        frame.time_base = Fraction(1, 1) / fps
        for pkt in ost.encode(frame):
            out.mux(pkt)
        n += 1
        if n % 60 == 0:
            print(f"\r    {n}/{frames or '?'} 帧  {n / max(time.time() - t0, 1e-6):.0f} fps",
                  end="", flush=True)

    for pkt in ost.encode(None):
        out.mux(pkt)
    out.close()
    inp.close()
    if n % 60:
        print(f"\r    {n}/{frames or '?'} 帧", end="")
    print(f"  用时 {time.time() - t0:.1f}s")
    return probe(dst)


def collect(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.mp4")))
        elif any(c in p for c in "*?["):
            files += sorted(glob.glob(p))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"跳过（不存在）: {p}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="mp4v -> 浏览器可播的 H.264 mp4")
    ap.add_argument("paths", nargs="+", help="文件、目录或通配符")
    ap.add_argument("--crf", type=int, default=23, help="质量 18~28，越小越清晰（默认 23）")
    ap.add_argument("--preset", default="medium", help="x264 preset（默认 medium）")
    ap.add_argument("--outdir", default=None, help="输出目录；不给则原地替换并备份原件")
    ap.add_argument("--force", action="store_true", help="已是 H.264 也重新转")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不转码")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        print("没有找到 mp4 文件")
        return 1

    todo = []
    for f in files:
        info = probe(f)
        ok = info["codec"] in ("avc1", "avc3") and info["faststart"]
        tag = "已是 H.264" if ok else f'{info["codec"]} / faststart={info["faststart"]}'
        if ok and not args.force:
            print(f"[跳过] {os.path.basename(f)}  ({tag})")
            continue
        print(f"[待转] {os.path.basename(f)}  ({tag}, {info['size'] / 1048576:.1f} MB)")
        todo.append(f)

    if args.dry_run or not todo:
        print(f"\n合计 {len(todo)} 个待转码" + ("（dry-run，未执行）" if args.dry_run else ""))
        return 0

    if not args.outdir:
        print("\n原件备份 -> 各目录下的 .mp4v_backup/")

    total_in = total_out = 0
    failed = []
    for i, f in enumerate(todo, 1):
        name = os.path.basename(f)
        print(f"\n[{i}/{len(todo)}] {name}")
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            dst = os.path.join(args.outdir, name)
            tmp = dst + ".part"
        else:
            tmp = f + ".part"
            dst = f
            backup_dir = os.path.join(os.path.dirname(os.path.abspath(f)), ".mp4v_backup")
            os.makedirs(backup_dir, exist_ok=True)
        try:
            src_size = os.path.getsize(f)
            info = transcode(f, tmp, crf=args.crf, preset=args.preset)
            if args.outdir:
                shutil.move(tmp, dst)
            else:
                shutil.move(f, os.path.join(backup_dir, name))
                shutil.move(tmp, dst)
            dst_size = os.path.getsize(dst)
            total_in += src_size
            total_out += dst_size
            print(f"    -> {info['codec']}  faststart={info['faststart']}  "
                  f"{src_size / 1048576:.1f} MB -> {dst_size / 1048576:.1f} MB")
        except Exception as e:  # noqa: BLE001
            failed.append((name, str(e)))
            print(f"    !! 失败: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    print("\n" + "=" * 60)
    print(f"成功 {len(todo) - len(failed)} / {len(todo)}"
          + (f"，总体积 {total_in / 1048576:.1f} MB -> {total_out / 1048576:.1f} MB"
             if total_in else ""))
    for name, err in failed:
        print(f"  失败 {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
