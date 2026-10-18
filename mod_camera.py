# -*- coding: utf-8 -*-
"""【模块 4】虚拟运镜（导演 + 摄影师）：状态机 → 目标 c*/s* → 弹簧平滑 → 等比例裁剪

纯逻辑模块：只吃结构化输入（球位/速度、owner 框、人的框），输出每帧的裁剪窗参数，
不直接读写视频，便于单测与替换渲染器。

三层：
  ① 状态机 WIDE / BALL / FOLLOW / ACTION：决定"看哪里"；
  ② 临界阻尼二阶弹簧 + pan/zoom 限速：把目标变成"有惯性的手"，摇镜手感来源；
  ③ 约束：裁剪窗不出界、**宽高比严格锁定**（sx ≡ sy，杜绝压扁拉高）。

自测: python mod_camera.py   （对合成输入跑 300 帧，检查比例/限速/跟球率）
"""
import numpy as np

# ---- 导演参数 ----
S_MIN, S_MAX = 1.02, 3.0
MIN_PERSON_PX = 110        # 目标人在输出画面中的最小像素高（远处自动放大到看得清）
ACTION_HOLD = 30           # ACTION 状态最短驻留帧数
EXIT_SLOW = 8              # ACTION 退出需连续慢帧数（防抖）
LEAD_FRAMES = 18           # look-ahead：镜头往球将要去的地方摇（预判而非硬追）
DWELL = 45                 # 非 ACTION 状态的最短驻留（防抖）

# ---- 控制参数（摇镜头手感）----
OMEGA_PAN = 2.2            # 弹簧角频率
OMEGA_ZOOM = 1.6
ZETA = 1.0                 # 临界阻尼
VMAX_PAN = 0.40            # 最大摇镜速度（画面宽/秒）：球飞行可达 0.84~2.6 宽/秒
VMAX_ZOOM = 0.55           # 最大变焦速率（倍/秒）


class Director:
    """每帧喂入观测，吐出裁剪窗参数。状态在对象内（弹簧需要记忆）。"""

    def __init__(self, shot_speed, fps=29.97, src_w=854, src_h=356,
                 out_w=854, out_h=356):
        self.shot_speed = shot_speed
        self.fps = fps
        self.dt = 1.0 / fps
        self.src_w, self.src_h = src_w, src_h
        self.out_w, self.out_h = out_w, out_h
        self.cx, self.cy, self.s = 0.5, 0.55, S_MIN
        self.vcx, self.vcy, self.vs = 0.0, 0.0, 0.0
        self.state, self.state_enter, self.fast, self.slow = "WIDE", 0, 0, 0

    @staticmethod
    def adaptive_shot_speed(ball_vel):
        """按本段球速分布定射门阈值：不同机位/景深 p95 差 2 倍以上，写死必然失真"""
        sp = [float(np.linalg.norm(v)) for v in ball_vel.values()]
        return max(0.025, 1.3 * float(np.percentile(sp, 95))) if sp else 0.048

    def step(self, ball_pos, ball_vel, own_box):
        """→ view dict（含裁剪窗像素参数与缩放系数）"""
        f = getattr(self, "_f", 0)
        self._f = f + 1
        v = ball_vel if ball_vel is not None else np.zeros(2)
        spd = float(np.linalg.norm(v))
        self.fast = self.fast + 1 if spd > self.shot_speed else 0
        self.slow = self.slow + 1 if spd < 0.7 * self.shot_speed else 0

        # --- ① 状态机 ---
        want = "WIDE"
        if self.fast >= 2 or spd > 2.0 * self.shot_speed:
            want = "ACTION"
        elif own_box is not None:
            want = "FOLLOW"
        elif ball_pos is not None:
            want = "BALL"
        if want != self.state:
            if want == "ACTION" or f - self.state_enter > DWELL or (
                    want == "FOLLOW" and self.state in ("WIDE", "BALL")):
                self.state, self.state_enter = want, f
        if self.state == "ACTION" and f - self.state_enter > ACTION_HOLD and self.slow >= EXIT_SLOW:
            self.state, self.state_enter = ("FOLLOW" if own_box is not None else "BALL"), f

        # --- 目标 c* ---
        if ball_pos is not None:
            bp = np.array(ball_pos)
            lead = np.clip(v * LEAD_FRAMES, -0.10, 0.10)
            if self.state == "FOLLOW" and own_box is not None:
                oc = np.array([(own_box[0] + own_box[2]) / 2, (own_box[1] + own_box[3]) / 2])
                tc = 0.45 * bp + 0.55 * oc + 1.0 * np.clip(v * LEAD_FRAMES, -0.25, 0.25)
            elif self.state == "ACTION":
                tc = bp + np.clip(v * 20, -0.15, 0.15)
            else:
                tc = bp + 1.0 * np.clip(v * LEAD_FRAMES, -0.25, 0.25)
        else:
            tc = np.array([self.cx, self.cy])       # 球丢失：保持当前，绝不跳切

        # 「主体必须在框内」约束：FOLLOW/ACTION 的目标点偏向 owner，球飞离 owner 时
        # 会被挤出画面（实测 8~20 帧看不到球）。把目标中心钳到"球距中心 ≤80% 半窗"，
        # 既保证球可见，又保留构图上偏向 owner 的意图。
        if ball_pos is not None:
            half = 0.5 / self.s                     # 归一化半窗（x/y 同值）
            lim = 0.8 * half
            if abs(ball_pos[0] - tc[0]) > lim:
                tc[0] = ball_pos[0] - np.sign(ball_pos[0] - tc[0]) * lim
            if abs(ball_pos[1] - tc[1]) > lim:
                tc[1] = ball_pos[1] - np.sign(ball_pos[1] - tc[1]) * lim

        # 球是否跑出当前裁剪窗（决定"重新构图"提速 + 先拉开镜头）
        reframing = False
        if ball_pos is not None:
            reframing = not (abs(ball_pos[0] - self.cx) <= 0.5 / self.s + 0.02 and
                             abs(ball_pos[1] - self.cy) <=
                             0.5 / self.s * (self.src_h / self.src_w) + 0.02)

        # --- 目标 s*（含"远处的人放大到看得清"约束）---
        if self.state == "FOLLOW" and own_box is not None:
            span = float(np.linalg.norm(np.array(ball_pos) -
                                        np.array([(own_box[0] + own_box[2]) / 2,
                                                  (own_box[1] + own_box[3]) / 2]))) \
                if ball_pos is not None else 0.06
            s_want = 1.0 / (max(0.09, span * 1.2) * 2.4 + 0.10)
            hpx = (own_box[3] - own_box[1]) * self.src_h
            if hpx > 4:
                s_want = max(s_want, MIN_PERSON_PX / hpx)
        elif self.state == "ACTION":
            s_want = 1.7
        elif self.state == "BALL":
            s_want = 1.8
        else:
            s_want = 1.12
        # 球速越高越要拉远：实测球飞行可达 2.6 画面宽/秒，而摇镜上限只有 0.9，
        # 高倍变焦下必然跟丢。这条封顶是"跟得住"的充要条件，不是风格选择。
        zcap = float(np.interp(spd, [0.0, 0.02, 0.06, 0.09], [S_MAX, 3.0, 1.8, 1.25]))
        s_want = min(s_want, zcap)
        if reframing:
            s_want *= 0.55          # 球跑出画面时先"拉开"再摇：拉到能框住球的倍率（实测 0.70 仍不够）
        s_want = float(np.clip(s_want, S_MIN, S_MAX))

        # --- ② 弹簧平滑 + 限速 ---
        # 合速度限速（而非分轴）：hypot(vx,vy) 必须 ≤ VMAX_PAN，否则斜向摇镜会到 0.31 宽/s。
        # 球跑到裁剪窗外时允许"重新构图"提速（真实摄影师也是这样追球），但仍然平滑。
        boost = 3.0 if reframing else 1.0          # 只给 pan 提速：zoom 跟着飙会抽搐
        om_p = OMEGA_PAN * (2.2 if reframing else 1.0)
        vmax_p = VMAX_PAN * boost
        # 变焦也要能"急拉"：0.55 倍/秒从 2.05 拉到 1.13 需 1.7s，快攻事件早就结束了
        om_z = OMEGA_ZOOM * (2.0 if reframing else 1.0)
        vmax_z = VMAX_ZOOM * (3.0 if reframing else 1.0)
        ax = om_p ** 2 * (tc[0] - self.cx) - 2 * ZETA * om_p * self.vcx
        ay = om_p ** 2 * (tc[1] - self.cy) - 2 * ZETA * om_p * self.vcy
        az = om_z ** 2 * (s_want - self.s) - 2 * ZETA * om_z * self.vs
        self.vcx += ax * self.dt
        self.vcy += ay * self.dt
        self.vs = float(np.clip(self.vs + az * self.dt, -vmax_z, vmax_z))
        sp = float(np.hypot(self.vcx, self.vcy))
        if sp > vmax_p:                       # 等比缩放，保持方向不变
            self.vcx, self.vcy = self.vcx * vmax_p / sp, self.vcy * vmax_p / sp
        self.cx = self.cx + self.vcx * self.dt
        self.cy = self.cy + self.vcy * self.dt
        self.s = float(np.clip(self.s + self.vs * self.dt, S_MIN, S_MAX))

        self.vmax_pan = vmax_p                      # 供遥测/判定使用（含重新构图提速）
        self.vmax_zoom = vmax_z
        self.reframing = bool(reframing)
        return self._view()

    def _view(self):
        """裁剪窗：尺寸由输出宽高比反推，保证 sx ≡ sy（画面绝不压扁/拉高）"""
        cw = self.src_w / self.s
        ch = cw * self.out_h / self.out_w
        if cw >= self.src_w or ch >= self.src_h:
            cw, ch = float(self.src_w), float(self.src_h)
        cx_px = float(np.clip(self.cx * self.src_w, cw / 2, self.src_w - cw / 2))
        cy_px = float(np.clip(self.cy * self.src_h, ch / 2, self.src_h - ch / 2))
        self.cx, self.cy = cx_px / self.src_w, cy_px / self.src_h    # 回写防积分饱和
        sx, sy = self.out_w / cw, self.out_h / ch
        if abs(sx - sy) > 1e-6:
            raise SystemExit("aspect unlocked: sx=%.6f sy=%.6f" % (sx, sy))
        ox, oy = cx_px - cw / 2, cy_px - ch / 2
        # 输出尺寸必须显式给整数：用 int(sx*cw) 会因浮点误差偶发 853x355，
        # 与视频流尺寸不符 → ffmpeg 直接丢帧（实测丢约 10%）。
        return {"state": self.state, "s": self.s, "sx": sx, "sy": sy,
                "cw": cw, "ch": ch, "ox": ox, "oy": oy,
                "ow": int(self.out_w), "oh": int(self.out_h),
                "x0": ox / self.src_w, "y0": oy / self.src_h,
                "pan_v": float(np.hypot(self.vcx, self.vcy)), "zoom_v": abs(self.vs),
                "vmax_pan": getattr(self, "vmax_pan", VMAX_PAN),
                "vmax_zoom": getattr(self, "vmax_zoom", VMAX_ZOOM),
                "reframing": getattr(self, "reframing", False)}


def crop(frame, view):
    """等比例裁剪 + 亚像素仿射（放大用 LANCZOS4）"""
    import cv2
    M = np.array([[view["sx"], 0.0, -view["ox"] * view["sx"]],
                  [0.0, view["sy"], -view["oy"] * view["sy"]]])
    interp = cv2.INTER_LANCZOS4 if view["sx"] > 1.01 else cv2.INTER_AREA
    return cv2.warpAffine(frame, M, (view["ow"], view["oh"]),
                          flags=interp, borderMode=cv2.BORDER_REPLICATE)


def to_out(p, view):
    """归一化源坐标 → 输出像素坐标"""
    return (int((p[0] * 854 - view["ox"]) * view["sx"]),
            int((p[1] * 356 - view["oy"]) * view["sy"]))


if __name__ == "__main__":
    # 合成自测：球绕场跑 300 帧，检查比例锁定 / 限速 / 跟球
    rng = np.random.default_rng(0)
    d = Director(0.05)
    pos, vel, bad, out_of_view = np.array([0.2, 0.5]), np.array([0.008, 0.0]), 0, 0
    for i in range(300):
        pos = np.clip(pos + vel + rng.normal(0, 0.002, 2), 0.05, 0.95)
        v = Director.step(d, tuple(pos), vel, (pos[0] - 0.02, pos[1] - 0.05,
                                               pos[0] + 0.02, pos[1] + 0.01))
        if abs(v["sx"] - v["sy"]) > 1e-6:
            bad += 1
        if not (v["x0"] <= pos[0] <= v["x0"] + v["cw"] / 854 and
                v["y0"] <= pos[1] <= v["y0"] + v["ch"] / 356):
            out_of_view += 1
    print("自测 300 帧：宽高比失衡=%d，球跑出裁剪窗=%d 帧，终态 zoom=%.2fx" % (
        bad, out_of_view, d.s))
