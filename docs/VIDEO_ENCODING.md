# 视频输出编码规范（生成视频时必须遵守）

## 一、核心规则

> **这个仓库产出的任何 .mp4，都必须是 H.264（fourcc `avc1`）+ yuv420p + moov 在文件头（faststart）。
> 不允许用 `cv2.VideoWriter_fourcc(*"mp4v")` 作为最终交付格式。**

原因：mp4v = MPEG-4 Part 2，Chrome / Edge / Firefox / Safari、飞书、网盘预览、评审页面一律不支持解码，
页面只会显示「该视频编码格式浏览器不支持在线预览，请下载后用本地播放器查看」。
Windows 版 OpenCV 缺 `openh264-*.dll`，写 `avc1` 会失败并**静默退回 mp4v**，肉眼看不出问题，交付时才暴露。

## 二、可直接粘贴给模型的提示词

```
生成/写出 mp4 视频时：
1. 禁止用 cv2.VideoWriter(..., fourcc("mp4v")) 产出最终文件；它是 MPEG-4 Part 2，
   浏览器、飞书、网盘预览都播不了，只会提示「编码格式不支持，请下载后本地播放」。
2. 必须用 H.264：fourcc 为 avc1，像素格式 yuv420p，moov 索引在文件头（+faststart），
   否则无法边下边播。
3. Windows 上 OpenCV 通常没有 openh264，写 avc1 会失败并静默退回 mp4v，
   不要只依赖 cv2.VideoWriter 的返回值，写完必须校验容器里的 fourcc。
4. 优先直接以 H.264 写出；已有的 mp4v 文件交付前统一转码。
5. 画面尺寸必须宽高都为偶数，否则 yuv420p 编码会失败。
6. 交付前自检：读 MP4 容器 stsd 里的 fourcc 应为 avc1，且 moov 位置在 mdat 之前。
```

## 三、本仓库的落地方式

直接输出（推荐）：

```python
from webvideo import open_writer          # 项目根目录，接口兼容 cv2.VideoWriter
vw = open_writer(out_path, fps, (W, H))   # 内部 PyAV + libx264 + yuv420p + faststart
vw.write(bgr_frame)
vw.release()
```

已有文件批量转码：

```bash
python web_mp4.py <目录或文件>            # 原地替换，原件备份到 .mp4v_backup/
python web_mp4.py <目录> --dry-run        # 先看哪些需要转
python web_mp4.py <目录> --crf 20         # 默认 23，实测 PSNR 约 40 dB
```

## 四、交付前自检

```bash
python -c "from web_mp4 import probe; import glob,sys; [print(f, probe(f)) for f in sorted(glob.glob(sys.argv[1]))]" "rf_out45/*.mp4"
```

合格标准：`codec == 'avc1'` 且 `faststart` 为 True。

实测参考：854x356 / 29.97fps / 510 帧的片段，mp4v 4.9 MB → H.264 2.9 MB，PSNR 39.9 dB。
