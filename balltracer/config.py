# BallTracer — 推理/跟踪默认超参（与验证通过的管线完全一致）
# 单位说明：坐标均为归一化值（相对 2x 上采样后画面，即 1708×712）；速度单位为「画面宽/帧」。

# ---- SAHI 切片 ----
NX = 3            # 切片列数
OV = 0.20         # 切片重叠比例
KEY_EVERY = 3     # I 帧间隔：每 KEY_EVERY 帧做一次全画面检测，其余为 P 帧（仅 ROI 切片）

# ---- 候选采集 ----
CONF = 0.03       # 候选采集置信阈值（刻意调低，靠后级时序门限过滤）
IMGSZ = 1280      # YOLO 推理尺寸

# ---- 双向时序门限（物理速度约束）----
GATE_BASE = 0.02
GATE_V = 1.5
GATE_MAX = 0.04      # 物理速度硬约束：球速 p95≈0.016/帧，0.04/帧 留 2.5x 余量
RECAP_STEP = 0.003  # 失锁后门限随丢失帧数线性放宽
MAXLOST = 25        # 最大连续失锁帧数，超过则链重置
INIT_CONF = 0.30    # 起轨所需最低置信（双向任一方向命中即建链）
CONF_MIN = 0.05     # 进入门限的候选最低置信
FILL_MAX = 15       # 最大插值补帧间隙
JUMP_HARD = 0.06    # 链路级跳变清理：相邻位移超过 1.8 宽/s 视为不可能跳变

# ---- 默认权重（相对仓库根；权重文件不在仓库内，见 .gitignore）----
DEFAULT_TILE_WEIGHTS = "runs/ball_tile_v3/weights/best.pt"   # 切片域最终检测模型
DEFAULT_FULL_WEIGHTS = "runs/ball11n_vb/weights/best.pt"     # 全帧 YOLO11n(COCO 微调)
DEFAULT_OLD_WEIGHTS = "runs/ball11n/weights/best.pt"         # 旧全帧基线

# ---- 输出视频规格（与源一致）----
OUT_W, OUT_H = 854, 356
FPS_FALLBACK = 29.97
