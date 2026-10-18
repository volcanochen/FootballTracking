# BallTracer v1.0.0 · 足踪（足球检测与轨迹预计算引擎）

> 本仓库的球检测 + 轨迹预计算逻辑已封装为 **`balltracer`** Python 包并打 tag **`v1.0.0`**。
> 一键运行：
> ```bash
> python -m balltracer --video full.mp4 --segs segs_cur.json --seg-ids 0,1,2,3,4,5,6,7 \
>   --tile-weights runs/ball_tile_v3/weights/best.pt
> ```
> 技术细节见 [`docs/TECHNICAL.md`](docs/TECHNICAL.md)。核心：SAHI 切片检测 + MPEG 式 I/P 帧分区推理 + 物理速度门限时序跟踪 + 长时轴跳变清理 + 插值补帧。

---

# VCNC 足球跟焦 · 球检测与轨迹预计算管线

> 仓库：github.com/volcanochen/h-player（`vcnc-football` 分支）
> 本目录：`_test/yolo_pipeline/` — PC 端（服务器侧）预计算管线，为 APK 端跟焦放大镜提供球轨迹。
> 视频素材：VCNC 五人场实况录像（854×356 低清源，固定俯拍机位，29.97fps）。
> 仅个人学习验证用，视频与权重不外传。

---

## 1. 目标与最终架构

目标：跟焦放大镜需要知道每一时刻"球在哪"。端侧自研检测在低清源上不可用（详见 §4 演进史），
最终方案是 **PC 预计算轨迹 → 推送到手机 → 端侧只做回放**：

```
┌─────────────────────────── PC（RTX 3080 Ti / 服务器侧）───────────────────────────┐
│                                                                                   │
│  full.mp4 ──► detect.py      YOLO11n(COCO) 2x+1280 → dets/<i>.json（person/ball）  │
│           ──► color_pass.py  青色候选（辅助，已证实含大量污染）→ dets_cyan/          │
│           ──► motion_pass.py |Δ(B−R)| 色度差分运动证据 → dets_mot/                 │
│           ──► pose_pass.py   yolo11n-pose 肩线朝向（实测无效，保留备用）→ dets_pose/ │
│           ──► train_ball.py  yolo11n 领域微调（白球）→ runs/ball11n/weights/best.pt│
│           ──► eval_ft.py     微调模型重跑 → dets_ft/ + dets_mot_ft/                │
│           ──► viterbi.py     全段 HMM 轨迹优化（关键帧锚定 + 物理补轨迹）           │
│                    │                                                              │
│                    ▼                                                              │
│              tracks.json（行为层：sceneStep/composeScene 复用 feed.js 原函数）      │
└──────────────────────────────┬────────────────────────────────────────────────────┘
                               │ adb push → /sdcard/Android/data/com.vcncfootball/files/tracks/
┌──────────────────────────────▼────────────────────────────────────────────────────┐
│  APK（vcnc-football）                                                              │
│  ProxyServer /api/tracks/<bvid>.json → feed.js loadExtTracks()                     │
│  prereadTrack 优先采用外部轨迹；双 deck 无缝连播 + 跟焦放大镜（CSS transform）        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**核心设计原则（与需求方确认）：**
1. 场上**只有一个球** —— 轨迹层天然单路径。
2. 高置信锁定后，后续帧**只考虑相邻位置**的候选（转移门限 GATE）。
3. 高速运动/遮挡时单帧不可判 —— 不强求每帧检出，**关键帧找对即可，中间按物理规律补轨迹**
   （Viterbi 的 ABS 虚态 + ≤15 帧运动学插值，smoothstep 缓动）。
4. 单帧检测的价值有限，跨帧一致性才是主判据。

---

## 2. 关键发现（血泪教训，务必先读）

| # | 发现 | 证据 |
|---|------|------|
| 1 | **画面 (0.630, 0.565) 处有一个固定青色光源**，26/27 个早期"球"标注都落在它上面，跨 11 分钟比赛纹丝不动 | 人工标注坐标统计；早期的"青色小球 100% 帧覆盖"全是它 |
| 2 | **真球是白色小球**（3~8px），YOLO 高分检出目检全是白球；另有深色球出现 | ballref_sheet.png 高分球样片 |
| 3 | 青色候选池被严重污染：蓝短裤/球袜/围栏青漆/出口牌/光源全部命中 | sheets/ 触表标注（491 候选仅 27"球"，后证实 26 个是光源） |
| 4 | B−G 通道真球与误检完全重叠，**颜色通道不可分** | analyze_bgr.py（真球 B-G 中位 −18，误检 −15.8） |
| 5 | 俯拍机位下身体朝向不可测：正/背对镜头的人鼻子投影在肩线上，faceConf≈0 | pose_pass.py 逐帧诊断，加入先验后路径 0/150 帧变化 |
| 6 | 人框 YOLO 检出不稳：16% 帧全漏、框常只到膝盖 → 人框惩罚必须跨帧传播 | build/dets 分析 |
| 7 | 伪标签必须先目检种子再传播：传播半径 0.03/±4 帧会漂到鞋/衣物 | 种子评审 66 个中剔 7 个；用户评审再剔 6 个种子 + 44 个传播框 |

---

## 3. 目录清单

```
yolo_pipeline/
├── 模型与权重
│   ├── yolo11n.onnx        COCO 预训练检测（来自 yolo_baseline）
│   ├── yolo11n-pose.onnx   姿态（肩线朝向实验）
│   ├── yolo11n.pt          COCO 预训练（微调起点）
│   └── runs/ball11n/       微调产物（weights/best.pt）
├── 检测层（python，逐段产出）
│   ├── detect.py           YOLO 2x+1280 → dets/<i>.json + frames/*.rgba
│   ├── color_pass.py       青色候选 → dets_cyan/（辅助源，含光源污染）
│   ├── motion_pass.py      |Δ(B−R)| 色度差分运动证据 → dets_mot/（含青色候选 mot）
│   ├── pose_pass.py        姿态朝向 → dets_pose/（实测无效，保留）
│   └── eval_ft.py          微调模型重跑 → dets_ft/ + dets_mot_ft/
├── 轨迹层
│   ├── viterbi.py          全段 HMM 全局轨迹优化（核心，见 §5）
│   ├── tracks.js           行为层：候选融合选球(旧) + sceneStep/composeScene → tracks.json
│   └── fill_gaps(内置)     ≤15 帧运动学插值（smoothstep）
├── 数据集构建（微调用）
│   ├── make_sheets.py      第一轮触表（青色+YOLO 候选 → 人工标注）
│   ├── make_sheets2.py     第二轮触表（白球候选 + YOLO ballAll）
│   ├── labels_manual.py    第一轮人工标注结果（26/27 后证实为光源，已废弃）
│   ├── seeds_v2.json       66 个 YOLO≥0.30 种子（评审后保留 53）
│   ├── review_result.json  用户网页评审导出的 50 个坏框
│   ├── build_dataset2.py   种子+传播 → dataset2/（82 正帧 + 88 负帧）
│   └── train_ball.py       yolo11n 微调（imgsz=1280, 60ep, batch=6, 3080 Ti）
├── 评审工具
│   ├── make_review_gallery.py  交互式评审页（base64 内嵌，勾选导出 JSON）
│   ├── review_gallery.html     ← 浏览器打开即可评审
│   ├── seed_review/            种子放大目检图
│   └── review_frames/          整帧画框图（静态版）
└── 可视化/验证
    ├── viterbi_check.py    轨迹 vs YOLO 一致性统计 + 全帧标注
    ├── viterbi_zoom.py     逐帧放大目检（红圈=选球 V/i，绿点=候选，橙框=YOLO）
    └── analyze_bgr.py      颜色通道可分性分析（结论：不可分）
```

---

## 4. 技术演进史（为什么走到今天这一步）

| 阶段 | 方案 | 结果 | 结论 |
|------|------|------|------|
| v0 端侧自研 | 背景减除（Wren 式运行高斯）→ 前景连通域 → 双路径选球 | 真实低清源上 3% 检出 | **端侧检测不可用**，改 PC 预计算 |
| v1 PC 贪心 | YOLO 2x+1280 + 青色候选时序融合 + 静止断链/黑名单 | "real 检出 97%"（实为光源！） | 融合规则救不了错误的候选池 |
| v2 Viterbi+运动证据 | 全段 HMM + 色度差分 + 人框传播 + 朝向先验 | 光源锁死被打破✓，但漂移到球袜/短裤 | 位置先验全部用尽 |
| v3 领域微调 | 人工评审种子 → 时间传播 → yolo11n 微调 → Viterbi | val mAP50=0.667，光源误锁 2.2%✓；但用户确认的球位没进轨迹，Viterbi 仍乱飘 | 检测层问题应在检测层解决 |
| v4 锚点局部跟踪（当前） | 用户确认球位作硬锚点 → 锚点间物理连续双向链 → 强/弱分级+诚实留空 | 留一法找回隐藏锚点 25/29=86%，不再瞬移/骑球袜 | **关键帧必须由人确认，跟踪只做局部连续** |

**根因分析**：这个场景的根本困难不是"跟踪"，而是"检测"——球只有 3~8px 且被大量同色杂物包围。
COCO 预训练的 YOLO 对这种域外目标召回率低（~30% 帧有检出），而手工颜色规则又无法区分球与杂物。
**解法只能是领域微调**：让检测器自己学会"白色小球 vs 白线/鞋/衣物/光源"的判别，跟踪层（Viterbi）负责时序连续性。

---

## 5. Viterbi 轨迹优化（时间连续性的核心）

把一整段的所有帧候选建成 HMM，全局求最优路径（Viterbi 算法，O(帧数×候选²)）：

```
状态：每帧的候选 {青色候选, YOLO球框} + 「无球」虚态 ABS
发射分 = 面积分(球尺度0.7/其他0.25) × 人框惩罚(中部0.05/近脚0.5/框外1.0)
         × 运动证据 mot_fac(0.15~1.0)   # |Δ(B−R)|：静止物≈0，移动球 3~8+
转移分 = −(位移/0.06)²            # 运动连续性，位移>0.14 直接禁止（瞬移）
         ABS↔候选 有 Init(1.0)/Miss(0.7) 惩罚  # 防止路径频繁跳进跳出
输出：全局最优路径 → real 帧(锚) + ≤15 帧缺口插值(物理补轨迹)
```

- **为什么优于贪心跟踪**：贪心每帧只看上一帧，锁错后靠事后规则补救；Viterbi 让"长而连贯的路径"
  天然得高分，孤立误检（接进来要付 2 次 Init 惩罚）和静止杂物（运动证据压分）全局出局。
- **单球约束**天然满足（单路径）。
- **"关键帧+物理补"**：ABS 虚态允许中间帧无球，两端锚点间按匀速+smoothstep 缓动插值。

---

## 5b. 锚点局部跟踪 track_v2.py（当前方案）

Viterbi 是全局寻优：门限内允许"换到另一条更好路径"，在候选稀疏段会瞬移乱飘，且**人工确认的球位只用于训练、未约束轨迹**（59 个确认锚点仅 2 个与 Viterbi 输出一致）。v4 按用户原则重写：

> 场上只有一个球；确认高置信球位后只考虑相邻位置；不要求每帧找到球，关键帧找对即可，中间按物理规律补轨迹。

```
人工确认球位 = 硬锚点（seeds_v2.json 通过评审的部分，绿圈 A）
      ▼
锚点间双向链式跟踪：速度 EMA 预测 → 门限内选候选 → 汇合
  门限自适应：静止 0.05 → R = 0.035 + 2.0·|v|（上限 0.10，杜绝瞬移）
  丢失再捕获：每步扩半径 0.006，上限 +0.05；MAXLOST=12 步断链
      ▼
强/弱分级（宁缺勿错）：
  强 = 微调模型 score≥0.30                          → 红圈 V
  弱 = 其余（青色候选一律弱：球袜/短裤 mot 1.5 也达标；加 mot≤10 上限，真球 3~8、衣物 16~46）→ 黄圈 W
  连续 8 步只有弱候选 → 断链留空
      ▼
短空洞（≤12 步）smoothstep 插值补齐（橙圈 i）；长空洞诚实留空
```

**验证（留一法 LOO）**：逐个隐藏锚点，让链条从相邻锚点自己找回，命中 25/29 = 86%。

**标注视频图例**：绿 A=人工确认锚点 / 红 V=强检测 / 黄 W=存疑弱候选 / 橙 i=插值 / 空白=留空。
生成：`make_anno_video.py`（轨迹按 30fps 平滑展开 + 18 帧运动尾巴）；抽查：`spotcheck.py`。

**教训**：低锚段（如 seg5 仅 1 锚点）曾让青色候选链连骑 70+ 帧蓝色球袜——弱候选没有硬锚约束时不可信，宁可留空。

---

## 6. 数据集与标注工作流（可复用的评审闭环）

```
YOLO≥0.30 球检出（剔光源/人框）          ← 种子生成（自动化）
      ▼
种子放大触表 seed_review/*.png           ← 人工逐个目检（我审 + 用户审）
      ▼  剔除误检种子（手套/球鞋/暗点）
时间传播：±4 采样帧，0.03 邻域白块继承     ← 自动（时间连续性）
      ▼
交互式评审页 review_gallery.html         ← 用户勾选（全对/全错/逐框），导出 JSON
      ▼  review_result.json（50 坏框）
build_dataset2.py → dataset2/            ← 82 正帧 + 88 负帧（坏框帧转硬负样本）
      ▼
train_ball.py → runs/ball11n/best.pt     ← 3080 Ti, ~40min
      ▼
eval_ft.py → dets_ft → viterbi → 对比验证 ← 不达标回到种子步骤迭代
```

评审页操作：点框号 chip 循环 未评→对→错；[全对] 整帧通过；[导出结果] 复制 JSON 发回。
进度存 localStorage，可分次评审。

---

## 7. 当前状态与下一步

**已完成：**
- [x] 数据集 v2.1（用户评审 50 坏框落地：82 正 + 88 负）→ 微调完成（val mAP50=0.667 / R=0.714），光源误锁率 ~100% → 2.2%
- [x] Viterbi 管线（含光源排除、运动证据、人框传播）
- [x] 交互式评审闭环（网页勾选 → JSON → 剔坏重建）
- [x] v4 锚点局部跟踪 track_v2.py（LOO 86%，标注视频 anno_s0~s7.mp4）

**进行中 / 下一步：**
- [ ] 用户评审新标注视频：s2/s3/s6（锚点多）应跟得住；确认框架后给低锚段（s0/s1/s4/s5/s7）出新一轮评审表补锚点
- [ ] 低锚段补锚点 → 重跑 track_v2 → 全段覆盖
- [ ] 生成 tracks.json 推真机（N6）做端到端跟焦验证
- [ ] 若召回不足：扩大种子、多场视频增广

**人物检测跟踪（下一阶段，方案已出）：**
- 详见 [`docs/PLAN_PERSON_SERVER.md`](docs/PLAN_PERSON_SERVER.md)（方案 A · 服务器预计算）、
  [`docs/PLAN_PERSON_DEVICE.md`](docs/PLAN_PERSON_DEVICE.md)（方案 B · 设备端实时）、
  [`docs/EVAL_PERSON.md`](docs/EVAL_PERSON.md)（共用评测协议：帧级 85%/5s）
- 已实测：yolo26n 裸模型在本域 imgsz=640 漏检 ~35% 的人，imgsz=1280 召回正常但会误检广告牌海报人物
  （目检图 `../probe_person_s*.jpg`，仓库外）

**自动导播（球 + 人 + owner + 虚拟运镜，四/五模块已解耦）：**
- 模块划分：`mod_ball.py`（球轨迹，吃 `ball_best.py` 每帧 5 次推理融合的候选）
  → `mod_person.py`（人检测 + ByteTrack 风格关联：恒速外推 + IoU + 球衣 HSV ReID + 匈牙利分配）
  → `mod_owner.py`（带球人：脚点距离 + 迟滞 + 换人门槛 + 8 帧时序投票）
  → `mod_camera.py`（连续变焦导演）/ `mod_two_shot.py`（**两档镜头**）
- 编排与判定：`auto_director.py` / `auto_two_shot.py` 出视频 + 逐帧遥测；
  `eval_director.py` / `eval_two_shot.py` 做 AI 判定（比例锁定 / 球无闪走 / owner 覆盖与抖动 /
  镜头跟球 / 变焦平滑 / 换档方向正确等 12~13 项）
- 两档镜头语义（`mod_two_shot.py`）：owner 在远半场 → 只取源画面 **0.75×0.75** 的区域、
  **等比放大填满全屏**（1/0.75 = 1.3333 倍，严格不变形）；回到近半场 → 还原 s=1.0 原始比例。
  窗口大小固定，位置默认跟随带球人（`ANCHOR="owner"`；`"tl"` 为钉死左上角的对照模式）。
- **关键实测**：`corr(脚点y, 人像高) = 0.96~0.99`，而 `corr(x, 人像高) = 0.66/0.89/0.49/-0.46`
  —— 侧视全景机位下"远近"必须用脚点 y 判，用 x 会把左侧近处误判成远处。
- 宽高比教训（用户反馈"不能压扁拉高"）：裁剪窗尺寸必须由**输出宽高比反推**并断言 `sx ≡ sy`，
  且输出尺寸显式取整，否则 `int(sx*cw)` 偶发 853×355 会令 ffmpeg 丢帧约 10%。

**已知限制：**
- 标注只有 8 个测试段（~2 分钟有效画面），换场地/光照需重新走标注闭环
- pose 朝向先验无效（俯拍机位），代码保留但 Viterbi 中贡献为 0
- 微调模型对"高速运动模糊"的球可能仍漏检——按设计交给 Viterbi 补轨迹，不追求单帧

---

## 8. 常用命令

```bash
PY=C:/Users/volcano/.workbuddy/binaries/python/envs/default/Scripts/python.exe

# 检测层（已算好 dets 0~7，新视频需重跑）
$PY detect.py segs_cur.json 0,1,2,3,4,5,6,7
$PY motion_pass.py

# 微调（数据集更新后）
$PY build_dataset2.py && $PY train_ball.py

# 微调模型评测 → Viterbi → 可视化
$PY eval_ft.py
VIT_DET=ft $PY viterbi.py
$PY viterbi_check.py && $PY viterbi_zoom.py

# v4 锚点局部跟踪（当前方案）→ 标注视频 → 抽查
$PY track_v2.py                      # 输出 dets_track/，含 LOO 留一验证
"C:/Users/volcano/AppData/Local/Programs/Python/Python313/python.exe" make_anno_video.py   # 需 cv2，用系统 Python
"C:/Users/volcano/AppData/Local/Programs/Python/Python313/python.exe" spotcheck.py

# 生成 tracks.json（行为层，node）
VIT=1 node tracks.js segs_cur.json    # 待接入 BALL_SRC=vit 分支

# 推送到真机（N6）
adb push tracks.json /sdcard/Android/data/com.vcncfootball/files/tracks/BV1fRae6LEmJ.json

# ── 自动导播（模块化）──
PY=.../python.exe
$PY ball_best.py 2,3,6,10            # 球：每帧 5 次推理融合的最佳候选 → cand_best_*.json
$PY person_dets.py 2,3,6,10          # 人：批量生成检测缓存 dets_person_cache_*.npy

# 连续变焦导演（WIDE/BALL/FOLLOW/ACTION 状态机 + 弹簧手感）
$PY auto_director.py 2,3,6,10         # → auto_director_s*.mp4 + telemetry_s*.json
$PY eval_director.py 2,3,6,10         # AI 判定 12 项

# 两档镜头（远=0.75×0.75 放大到全屏 / 近=原始比例）
$PY auto_two_shot.py 2,3,6,10         # → two_shot_s*.mp4 + telemetry_two_s*.json
$PY eval_two_shot.py 2,3,6,10         # AI 判定 13 项
```

## 9. 相关分支与产物

- `main`：H-Player 通用 B 站集锦播放器（v1.0.0 已发布）
- `vcnc-football`：本实验分支，产物 VCNC-Football.apk（未发布）
- 端侧 UI：左下角控制条（播放/暂停 + 0.5/0.25/0.125 慢动作）、左滑亮度/右滑音量、单击暂停、本场播完停住
- 构建链：免 Gradle 离线 aapt2 → javac → d8 → zipalign → apksigner（android/build_apk.sh）
