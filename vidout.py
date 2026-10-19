# -*- coding: utf-8 -*-
"""H.264 MP4 写出器 + 容器自检（项目硬性要求，所有视频产出必须走这里）

为什么不用 cv2.VideoWriter：
  1. OpenCV 的 fourcc("mp4v") 产出的是 MPEG-4 Part 2，浏览器/飞书/网盘预览一律播不了，
     只会提示「编码格式不支持，请下载后本地播放」；
  2. Windows 上 OpenCV 通常没编 openh264，写 fourcc("avc1") 不会报错，而是**静默退回
     mp4v** —— 只看 VideoWriter 的返回值根本发现不了；
  3. 所以这里直接用 PyAV 调 libx264，并在写完/交付前强制解析 MP4 容器校验。

硬性要求（逐条对应）：
  ① 最终文件必须是 H.264：MP4 容器 stsd 里的 fourcc 应为 avc1；
  ② pix_fmt 必须是 yuv420p（否则播放器/浏览器不解）；
  ③ moov 索引必须在 mdat 之前（+faststart），否则无法边下边播；
  ④ 宽高必须都为偶数，否则 yuv420p 直接编码失败；
  ⑤ 已有 mp4v 文件交付前用 transcode_to_h264() 统一转码。

用法:
    from vidout import H264Writer, check_mp4
    w = H264Writer("out.mp4", 854, 356, 29.97)
    for f in frames: w.write(f)
    w.close()                      # 内部已校验，失败直接抛异常
    ok, info = check_mp4("out.mp4")
"""
import os

import av
from fractions import Fraction


def _iter_boxes(data):
    """遍历 MP4 top-level box → [(type, offset, size), ...]"""
    out, i, n = [], 0, len(data)
    while i + 8 <= n:
        size = int.from_bytes(data[i:i + 4], "big")
        typ = data[i + 4:i + 8].decode("latin1")
        hdr = 8
        if size == 1:                                  # 64 位扩展尺寸
            size = int.from_bytes(data[i + 8:i + 16], "big")
            hdr = 16
        elif size == 0:                                # 延伸到文件尾
            size = n - i
        if size < hdr or i + size > n:
            break
        out.append((typ, i, size))
        i += size
    return out


def check_mp4(path):
    """解析 MP4 容器自检 → (ok, info)

    校验四项：codec=h264 / pix_fmt=yuv420p / stsd fourcc=avc1 / moov 在 mdat 之前。
    """
    info = {"path": os.path.basename(path)}
    c = av.open(path)
    try:
        if not c.streams.video:
            return False, {"err": "无视频流"}
        st = c.streams.video[0]
        cc = st.codec_context
        info.update({"codec": cc.name, "pix_fmt": getattr(cc.pix_fmt, "name", str(cc.pix_fmt)),
                     "w": cc.width, "h": cc.height,
                     "fps": float(st.average_rate or 0),
                     "n": st.frames or None,
                     "dur": float(st.duration * st.time_base) if st.duration else None})
    finally:
        c.close()

    with open(path, "rb") as fh:
        data = fh.read()
    boxes = _iter_boxes(data)
    t = {b[0]: b for b in boxes}
    moov, mdat = t.get("moov"), t.get("mdat")
    info["moov_off"] = moov[1] if moov else None
    info["mdat_off"] = mdat[1] if mdat else None
    info["faststart"] = bool(moov and mdat and moov[1] < mdat[1])

    fourcc = None
    if moov:
        seg = data[moov[1]:moov[1] + moov[2]]
        k = seg.find(b"stsd")
        if k >= 0 and k + 20 <= len(seg):
            # stsd: 'stsd'(4) ver+flags(4) entry_count(4) entry_size(4) fourcc(4)
            #      → fourcc 位于 stsd 起点 +16
            fourcc = seg[k + 16:k + 20].decode("latin1")
    info["fourcc"] = fourcc

    bad = []
    if info["codec"] != "h264":
        bad.append("codec=%s" % info["codec"])
    if info["pix_fmt"] != "yuv420p":
        bad.append("pix_fmt=%s" % info["pix_fmt"])
    if fourcc != "avc1":
        bad.append("fourcc=%s" % fourcc)
    if not info["faststart"]:
        bad.append("moov 不在 mdat 前")
    if info["w"] % 2 or info["h"] % 2:
        bad.append("尺寸非偶数 %dx%d" % (info["w"], info["h"]))
    info["problems"] = bad
    return (not bad), info


class H264Writer:
    """逐帧写入 H.264/yuv420p/faststart 的 MP4。close() 时自动校验，不合格直接抛异常。"""

    def __init__(self, path, w, h, fps, crf=20, preset="medium"):
        if w % 2 or h % 2:
            raise ValueError("宽高必须都为偶数（yuv420p 要求）：%dx%d" % (w, h))
        self.path, self.w, self.h, self.fps = path, int(w), int(h), float(fps)
        self.container = av.open(path, "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264",
                                                rate=Fraction(int(round(fps * 1000)), 1000))
        self.stream.width, self.stream.height = int(w), int(h)
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": preset,
                               "profile": "high", "level": "4.0"}
        self._i = 0
        self.closed = False

    def write(self, frame_bgr):
        import cv2
        if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
            raise ValueError("帧尺寸 %s 与写出器 %dx%d 不符" % (
                frame_bgr.shape[:2], self.w, self.h))
        f = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")
        f.pts = self._i
        self._i += 1
        for pkt in self.stream.encode(f):
            self.container.mux(pkt)

    def close(self):
        if self.closed:
            return
        for pkt in self.stream.encode(None):        # flush 编码器
            self.container.mux(pkt)
        self.container.close()
        self.closed = True
        ok, info = check_mp4(self.path)
        if not ok:
            raise RuntimeError("H.264 自检未过：%s → %s" % (self.path, info["problems"]))
        return info


def transcode_to_h264(src, dst, crf=20, preset="medium"):
    """把已有（多半是 mp4v 的）MP4 统一转码成 H.264/yuv420p/faststart。"""
    ic = av.open(src)
    ist = ic.streams.video[0]
    w, h = ist.codec_context.width, ist.codec_context.height
    if w % 2:
        w -= 1                                       # 奇数宽高裁成偶数，否则 yuv420p 失败
    if h % 2:
        h -= 1
    oc = av.open(dst, "w", options={"movflags": "+faststart"})
    ost = oc.add_stream("libx264", rate=ist.average_rate or Fraction(30, 1))
    ost.width, ost.height = w, h
    ost.pix_fmt = "yuv420p"
    ost.options = {"crf": str(crf), "preset": preset, "profile": "high", "level": "4.0"}
    n = 0
    for frame in ic.decode(ist):
        if frame.width != w or frame.height != h:
            frame = frame.reformat(width=w, height=h)
        frame = frame.reformat(format="yuv420p")
        for pkt in ost.encode(frame):
            oc.mux(pkt)
        n += 1
    for pkt in ost.encode(None):
        oc.mux(pkt)
    ic.close()
    oc.close()
    ok, info = check_mp4(dst)
    if not ok:
        raise RuntimeError("转码自检未过：%s → %s" % (dst, info["problems"]))
    return info, n


if __name__ == "__main__":
    import tempfile

    import cv2
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.mp4")
    wr = H264Writer(p, 854, 356, 29.97)
    for i in range(30):
        img = np_full = __import__("numpy").zeros((356, 854, 3), __import__("numpy").uint8)
        cv2.putText(img, "frame %d" % i, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2)
        wr.write(img)
    info = wr.close()
    print("自写自检:", info)
    ok2, i2 = check_mp4(p)
    print("复检:", ok2, i2["fourcc"], i2["pix_fmt"], i2["faststart"])
    assert ok2 and i2["fourcc"] == "avc1" and i2["pix_fmt"] == "yuv420p" and i2["faststart"]

    # 对照：cv2.VideoWriter 的 mp4v 产物应当被判定为不合格
    p2 = os.path.join(d, "cv.mp4")
    vw = cv2.VideoWriter(p2, cv2.VideoWriter_fourcc(*"mp4v"), 29.97, (854, 356))
    for i in range(30):
        vw.write(img)
    vw.release()
    ok3, i3 = check_mp4(p2)
    print("cv2 mp4v 对照:", ok3, i3["fourcc"], i3["pix_fmt"], i3["problems"])
