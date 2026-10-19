# -*- coding: utf-8 -*-
"""【模块 5】两档镜头（近 / 远）：owner 在远半场 → 平缓推近；回到近半场 → 还原原始比例

只保留两种镜头：
  FAR  —— **同时发现球和带球人、且带球人进入远半场**（脚点 y < 半场线）时：
           只取源画面左上角的 0.75×0.75 区域，等比放大填满全屏
           （1/0.75 = 1.3333 倍），窗口位置**钉死左上角**，画面稳定不晃；
  NEAR —— 其余情况（带球人回到近半场，或球/带球人任一丢失）：
           回到 S_NEAR = 1.0，即**原始比例，一点都不放大**。

为什么用"脚点 y"当远近判据（而不是 x）：
  实测 4 段 owner 的 corr(脚点y, 人像高) = 0.96~0.99，而 corr(x, 人像高) 只有
  0.66/0.89/0.49/-0.46 —— 这是标准侧视全景机位，画面越靠下离摄像机越近，
  x 只代表沿边线方向的位置，与距离无关。用 x 判远近会把"左侧近处"误判成远处。

三段式设计：
  ① 标定 calibrate()：从本段数据自适应出"半场线 y_split"（全场人脚点中位数）；
  ② 档位决策：迟滞带 BAND + 最短驻留 DWELL，杜绝在半场线附近反复横跳；
  ③ 推拉：定时长 smoothstep 缓动（C1 连续、无过冲、时长可控），比弹簧更适合
     "只有两档"这种离散目标 —— 弹簧会留下指数尾巴，观感上不像摄影师，
     更像没停稳。摇镜用一阶低通 + 限速（ANCHOR="tl" 时目标恒为左上角）。

输出 view dict 与 mod_camera 同构，可复用 crop() / to_out() / eval。

自测: python mod_two_shot.py
"""
import numpy as np

# ---- 远档：只取源画面左上角 0.75×0.75，放大填满全屏 ----
FAR_WIN = 0.75           # 远档裁剪窗占源画面的比例（宽高都是 0.75 倍）
S_NEAR = 1.0             # 近档：原始比例（严格 1.0，不裁剪不放大）
S_FAR = 1.0 / FAR_WIN    # = 1.3333；窗与输出同宽高比 → 放大到全屏严格不变形
ANCHOR = "tl"            # 窗口位置："tl" 钉死左上角（默认）/ "owner" 跟随带球人

# ---- 档位决策 ----
BAND = 0.025             # 半场线迟滞带（归一化 y）：带内不切换
DWELL = 20               # 最短驻留帧数（0.67s）：再快就是抽搐不是运镜
RAMP_SEC = 1.4           # 推拉时长：1.0↔1.333 走 1.4 秒，慢到看不出机械感

# ---- 摇镜跟随 ----
TAU_PAN = 0.25           # 一阶低通时间常数（秒）
VMAX_PAN = 0.50          # 最大摇镜速度（画面宽/秒）


def calibrate(all_foot_ys):
    """从本段数据自适应标定半场线 → y_split

    全场所有人脚点 y 的中位数 ≈ 场地"中线"在画面里的位置（人近似均布）。
    远档倍率是固定的 1/0.75，不需要也不应该自适应 —— 用户要的就是"四分之三"。
    """
    return float(np.median(all_foot_ys)) if all_foot_ys else 0.42


class TwoShotDirector:
    """每帧喂 owner 框，吐裁剪窗。只有 FAR / NEAR 两档。"""

    def __init__(self, y_split, fps=29.97, src_w=854, src_h=356,
                 out_w=854, out_h=356):
        self.y_split = float(y_split)
        self.fps = fps
        self.dt = 1.0 / fps
        self.src_w, self.src_h = src_w, src_h
        self.out_w, self.out_h = out_w, out_h
        self.s_far = S_FAR
        self.cx, self.cy = FAR_WIN / 2, FAR_WIN / 2
        self.s = S_NEAR
        self.shot, self.shot_enter = "NEAR", 0
        self.ramp_f0, self.ramp_s0, self.ramp_s1 = 0, S_NEAR, S_NEAR
        self._f = 0
        # 推拉速率上限 = smoothstep 导数峰值 1.5 × 总行程 / 时长
        self.vmax_zoom = 1.5 * abs(S_FAR - S_NEAR) / RAMP_SEC
        self.ramp_frames = max(1, int(RAMP_SEC * fps))

    def _set_shot(self, want, f):
        # 推拉没走完不许改主意：摄影师一旦按下推拉键就推到底，中途反悔会变成抽搐，
        # 而且会让"档位"和"人实际在哪个半场"对不上（实测 seg2 语义一致率只有 0.91）。
        if want == self.shot or f - self.shot_enter < DWELL:
            return
        if f - self.ramp_f0 < self.ramp_frames and self.ramp_s1 != self.ramp_s0:
            return
        self.shot, self.shot_enter = want, f
        # 从"当前倍率"重新起坡：跨越式改主意也不会跳变
        self.ramp_f0, self.ramp_s0 = f, self.s
        self.ramp_s1 = S_FAR if want == "FAR" else S_NEAR

    def step(self, own_box, ball_pos=None):
        """own_box: 归一化 (x0,y0,x1,y1)；无 owner 时保持当前档位与画面。"""
        f = self._f
        self._f = f + 1

        # --- ① 档位决策（迟滞 + 驻留）---
        # 进入 FAR 的充要条件：**球和带球人同时被发现**，且带球人已进入远半场。
        # 缺任何一个都回 NEAR —— 没有"带球人"就没有推近的理由（球在飞行/
        # 带球人漏检时特写一块草皮毫无意义）。回退由 DWELL + "推拉未走完不许
        # 反悔"共同缓冲，不会因单帧丢球而抽搐。
        if own_box is None or ball_pos is None:
            want = "NEAR"
        else:
            y = float(own_box[3])                 # 脚点 y：小 = 远，大 = 近
            if y < self.y_split - BAND:
                want = "FAR"
            elif y > self.y_split + BAND:
                want = "NEAR"
            else:
                want = self.shot                  # 迟滞带内：保持不变
        self._set_shot(want, f)

        # --- ② 推拉：定时长 smoothstep ---
        u = (f - self.ramp_f0) / float(self.ramp_frames)
        u = 0.0 if u < 0 else (1.0 if u > 1 else u)
        u = u * u * (3.0 - 2.0 * u)               # C1 连续，首尾速度为 0
        s_new = self.ramp_s0 + (self.ramp_s1 - self.ramp_s0) * u
        zoom_v = abs(s_new - self.s) / self.dt
        self.s = float(s_new)
        # 推拉是否走完：摇镜低通比变焦收敛慢（tau=0.25s ≈ 需 36 帧才贴到左上角），
        # 所以"窗口已钉死"只能在这一帧之后判定，否则会把收敛过程误判成违规。
        self.settled = (f - self.ramp_f0) >= self.ramp_frames

        # --- ③ 摇镜 ---
        if ANCHOR == "tl":
            # 钉死左上角：目标恒定，直接赋值而不走低通 —— 一阶低通有指数尾巴，
            # 实测 42 帧后仍差 0.00047（≈0.4px），"钉死"永远差一点。
            # 且 cx=cy=0.375 时，_view() 的 clip 会把窗口顶到源画面左上角，
            # 对任意 s（含过渡中的 s）都成立，所以过渡依然连续。
            self.cx = self.cy = FAR_WIN / 2
            self.pan_v = 0.0
        else:
            if own_box is not None:
                tc = np.array([(own_box[0] + own_box[2]) / 2.0,
                               (own_box[1] + own_box[3]) / 2.0])
            elif ball_pos is not None:
                tc = np.array(ball_pos)           # 没 owner 时至少跟球
            else:
                tc = np.array([self.cx, self.cy])  # 什么都没有：原地不动
            a = 1.0 - float(np.exp(-self.dt / TAU_PAN))
            delta = tc - np.array([self.cx, self.cy])
            if float(np.linalg.norm(delta)) < 2e-4:
                self.cx, self.cy = float(tc[0]), float(tc[1])   # 死区内直接吸附
                self.pan_v = 0.0
            else:
                d = delta * a
                m = float(np.linalg.norm(d))
                lim = VMAX_PAN * self.dt
                if m > lim:
                    d = d * (lim / m)
                    m = lim
                self.cx = self.cx + float(d[0])
                self.cy = self.cy + float(d[1])
                self.pan_v = m / self.dt
        self.zoom_v = zoom_v
        return self._view()

    def _view(self):
        """裁剪窗由输出宽高比反推，sx ≡ sy（画面绝不压扁/拉高）"""
        cw = self.src_w / self.s
        ch = cw * self.out_h / self.out_w
        if cw >= self.src_w or ch >= self.src_h:
            cw, ch = float(self.src_w), float(self.src_h)
        cx_px = float(np.clip(self.cx * self.src_w, cw / 2, self.src_w - cw / 2))
        cy_px = float(np.clip(self.cy * self.src_h, ch / 2, self.src_h - ch / 2))
        self.cx, self.cy = cx_px / self.src_w, cy_px / self.src_h   # 回写防漂移
        sx, sy = self.out_w / cw, self.out_h / ch
        if abs(sx - sy) > 1e-6:
            raise SystemExit("aspect unlocked: sx=%.6f sy=%.6f" % (sx, sy))
        ox, oy = cx_px - cw / 2, cy_px - ch / 2
        return {"state": self.shot, "s": self.s, "sx": sx, "sy": sy,
                "cw": cw, "ch": ch, "ox": ox, "oy": oy,
                "ow": int(self.out_w), "oh": int(self.out_h),
                "x0": ox / self.src_w, "y0": oy / self.src_h,
                "pan_v": round(float(getattr(self, "pan_v", 0.0)), 5),
                "zoom_v": round(float(getattr(self, "zoom_v", 0.0)), 5),
                "vmax_pan": VMAX_PAN, "vmax_zoom": self.vmax_zoom,
                "reframing": False, "y_split": self.y_split, "s_far": self.s_far,
                "anchor": ANCHOR, "far_win": FAR_WIN, "settled": bool(self.settled)}


if __name__ == "__main__":
    # 自测 1：owner 从远端(0.30)走到近端(0.50)再走回去，且 150~200 帧球丢失
    y_split = calibrate([0.30 + 0.002 * i for i in range(60)])
    d = TwoShotDirector(y_split)
    bad, shots, zooms, far_steady, out_of_win, settled_n = 0, [], [], 0, 0, 0
    for i in range(400):
        y = 0.50 - 0.20 * (i / 200.0) if i < 200 else 0.30 + 0.20 * ((i - 200) / 200.0)
        bp = None if 150 <= i < 200 else (0.50, y - 0.02)     # 中间 50 帧模拟丢球
        v = d.step((0.45, y - 0.14, 0.55, y), bp)
        if abs(v["sx"] - v["sy"]) > 1e-6:
            bad += 1
        if not (0.0 <= v["x0"] and v["x0"] + v["cw"] / 854 <= 1.0 + 1e-6 and
                0.0 <= v["y0"] and v["y0"] + v["ch"] / 356 <= 1.0 + 1e-6):
            bad += 1
        if v["state"] == "FAR" and abs(v["s"] - S_FAR) < 0.002:
            far_steady += 1
            if v["settled"]:
                settled_n += 1
            # 远档稳态时，裁剪窗必须是源画面的 0.75×0.75。
            # 容差按 cw = 1/s 换算：dcw = ds/s²（s=1.333 时 ds=0.002 → dcw≈0.0011），
            # 直接用 1e-6 会把 smoothstep 的收敛尾巴误判成违规。
            if abs(v["cw"] / 854 - FAR_WIN) > 0.0015 or abs(v["ch"] / 356 - FAR_WIN) > 0.0015:
                bad += 1
            # 默认模式窗口必须钉死左上角。一阶低通是指数趋近，残差约 0.0006
            # （≈0.5px），故容差取 1e-3 而非 1e-6。
            if ANCHOR == "tl" and v["settled"] and (
                    abs(v["x0"]) > 1e-6 or abs(v["y0"]) > 1e-6):
                bad += 1
            # 记录"钉死左上角"的代价：owner 横向跑出 0.75 窗口的比例
            out_x = 0.45 + 0.05 < v["x0"] or 0.45 + 0.05 > v["x0"] + FAR_WIN
            if out_x:
                out_of_win += 1
        shots.append(v["state"])
        zooms.append(v["s"])
    sw = sum(1 for i in range(1, len(shots)) if shots[i] != shots[i - 1])
    print("自测A 400 帧：违规=%d  档位切换=%d 次  zoom=%.2f~%.2f  远档稳态=%d 帧" % (
        bad, sw, min(zooms), max(zooms), far_steady))
    print("       远档裁剪窗 = %.3f×%.3f 个源画面 → 放大 %.4fx 填满全屏（等比）" % (
        FAR_WIN, FAR_WIN, S_FAR))
    print("       tl 模式下 owner 横向出窗 %d/%d 帧（%.0f%%）—— 这是「画面钉死不晃」的既定代价" % (
        out_of_win, far_steady, 100.0 * out_of_win / max(1, far_steady)))
    print("       推拉已走完（settled）的远档稳态帧：%d/%d" % (settled_n, far_steady))
    # 丢球期间（150~200 帧）不应停留在 FAR
    mid = shots[160:200]
    print("       丢球窗口内档位集合=%s（应为 {'NEAR'}，即无球则回原始比例）" % sorted(set(mid)))

    # 自测 2：ANCHOR="owner" 对照模式
    import mod_two_shot as m
    m.ANCHOR = "owner"
    d2 = m.TwoShotDirector(y_split)
    xs = []
    for i in range(60):
        bx = 0.2 + 0.6 * (i / 59.0)          # owner 横向从 0.2 走到 0.8
        xs.append(d2.step((bx - 0.05, 0.10, bx + 0.05, 0.26), (bx, 0.24))["x0"])
    print("自测B owner 模式：owner 横移 0.2→0.8，窗口左边界 %.4f~%.4f（应随之变化）" % (
        min(xs), max(xs)))
    m.ANCHOR = "tl"
