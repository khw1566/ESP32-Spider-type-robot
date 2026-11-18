"""목표 좌표 조건부 거미형 4족 로봇 Gymnasium 환경.

관측은 실물 로봇에서 얻을 수 있는 값만 사용한다.
  - IMU(MPU6050): 몸체 기준 중력 방향 3 + 각속도 3 (노이즈·바이어스)
  - 관절각: 명령(개루프 추정) 각도 8  ← 실제 각도가 아님(탈조·유격 오차는 관측되지 않음)
  - 이전 행동 8
  - 거리센서 4 (수평 0/±45°, 전방 하향), 유효거리 200 cm, 쿨타임·지연·실패
  - 목표: 추정 위치 기준 거리, 방향 cos/sin, 목표 존재 플래그 4
  - 보행 클럭 sin/cos 2 (선택)
행동: 관절 8개 목표각 (정규화 -1~1)
"""
from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import load_config, deep_update
from .model_builder import JOINT_NAMES, LEG_NAMES, build_mjcf, output_torque
from .terrain import ArenaManager

REWARD_TERMS = ["progress", "reach", "heading", "upright", "height", "stand_bonus", "alive",
                "body_contact", "obstacle_contact", "energy", "torque", "action_rate",
                "joint_limit", "feet_air_time", "fall"]


def _lerp(lo, hi, t):
    return lo + (hi - lo) * t


class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: dict | None = None, stage: int = 0, progress: float = 0.0,
                 render_mode: str | None = None, record: bool = False, fixed_stage: bool = False):
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        c = self.cfg
        self.render_mode = render_mode
        self.record = record
        self.fixed_stage = fixed_stage
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(c))
        self.data = mujoco.MjData(self.model)
        self.arena = ArenaManager(self.model, c)
        m = self.model

        self.n_sub = c["sim"]["substeps"]
        self.dt_ctrl = 1.0 / c["sim"]["control_hz"]
        self.half = c["sim"]["arena_size"] / 2

        # ---- 인덱스 ----
        self.torso = m.body("torso").id
        self.jnt_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINT_NAMES])
        self.jnt_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINT_NAMES])
        rng_rad = np.array([m.jnt_range[m.joint(j).id] for j in JOINT_NAMES])  # 컴파일 후 rad
        self.j_lo, self.j_hi = rng_rad[:, 0], rng_rad[:, 1]
        contiguous = lambda a: len(a) == 8 and np.all(np.diff(a) == 1)
        self._qslice = slice(self.jnt_qadr[0], self.jnt_qadr[0] + 8) if contiguous(self.jnt_qadr) else self.jnt_qadr
        self._dslice = slice(self.jnt_dadr[0], self.jnt_dadr[0] + 8) if contiguous(self.jnt_dadr) else self.jnt_dadr
        self.foot_geoms = np.array([m.geom(f"{l}_foot").id for l in LEG_NAMES])
        robot_geoms = set(np.where(m.geom_contype == 2)[0].tolist())
        self.robot_body_geoms = robot_geoms - set(self.foot_geoms.tolist())
        self.floor_geom = m.geom("floor").id
        self.step_geoms = {m.geom(f"step{i}_g").id for i in range(len(self.arena.step_g))}
        world_geoms = set(np.where(m.geom_contype == 1)[0].tolist())
        self.obstacle_geoms = world_geoms - {self.floor_geom} - self.step_geoms
        self.foot_lookup = {g: i for i, g in enumerate(self.foot_geoms)}
        self.site_front = m.site("dist_front").id
        self.site_down = m.site("dist_down").id
        self.gyro_adr = m.sensor_adr[m.sensor("imu_gyro").id]
        self.ray_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid = np.zeros(1, dtype=np.int32)

        # ---- 행동 매핑 ----
        a = c["action"]
        self.act_nominal = np.radians(np.array([a["nominal_hip_deg"], a["nominal_knee_deg"]] * 4))
        self.act_scale = np.radians(np.array([a["hip_scale_deg"], a["knee_scale_deg"]] * 4))

        # ---- 랜덤화용 원본값 ----
        self.nom_body_mass = m.body_mass.copy()
        self.nom_body_ipos = m.body_ipos.copy()
        self.nom_geom_friction = m.geom_friction.copy()
        self.nom_gainprm = m.actuator_gainprm.copy()
        self.nom_biasprm = m.actuator_biasprm.copy()
        self.tau0 = output_torque(c["robot"])

        # ---- 공간 정의 ----
        self.frame_dim = 3 + 3 + 8 + 8 + 4 + 4 + (2 if c["obs"]["use_clock"] else 0)
        self.hist = c["obs"]["history"]
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.frame_dim * self.hist,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)

        self.stage = stage
        self.progress = progress
        self.stages = c["curriculum"]["stages"]
        self._renderer = None
        self.trace = []

    # ==================================================================
    # 커리큘럼 인터페이스 (VecEnv.env_method 로 호출)
    # ==================================================================
    def set_curriculum(self, stage: int, progress: float):
        self.stage = int(min(stage, len(self.stages) - 1))
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def get_curriculum(self):
        return self.stage, self.progress

    def _episode_params(self):
        cur = self.curriculum_cfg = self.cfg["curriculum"]
        st = self.stage
        prog = self.progress
        if (not self.fixed_stage) and st > 1 and self.np_random.random() < cur["replay_prob"]:
            st = int(self.np_random.integers(1, st))   # 이전 단계 재학습 (일어서기는 매 에피소드에 포함)
            prog = 1.0
        sc = self.stages[st]
        p = {"stage": st, "name": sc["name"], "progress": prog}
        p["rand"] = _lerp(*sc["randomization"], prog)
        lo, hi = sc["obstacles"]
        n_max = int(round(_lerp(lo, hi, prog)))
        p["n_obstacles"] = int(self.np_random.integers(lo, max(lo, n_max) + 1)) if hi > 0 else 0
        hlo, hhi = sc["step_height"]
        p["step_height"] = float(self.np_random.uniform(hlo, _lerp(hlo, hhi, prog))) if hhi > 0 else 0.0
        p["goal"] = sc["goal"]
        p["max_steps"] = int(sc["episode_seconds"] * self.cfg["sim"]["control_hz"])
        p["prone"] = self.np_random.random() < sc.get("init_prone_prob", 1.0)
        p["success_progress"] = sc.get("success_progress")
        p["weights"] = deep_update(self.cfg["reward"], sc.get("reward", {}))
        return p

    # ==================================================================
    def _randomize_physics(self, lvl: float):
        m, r, rng = self.model, self.cfg["randomization"], self.np_random
        m.body_mass[:] = self.nom_body_mass
        m.body_ipos[:] = self.nom_body_ipos
        m.geom_friction[:] = self.nom_geom_friction
        m.actuator_gainprm[:] = self.nom_gainprm
        m.actuator_biasprm[:] = self.nom_biasprm
        if lvl > 0:
            m.body_mass[self.torso] *= 1 + rng.uniform(-1, 1) * r["body_mass_frac"] * lvl
            m.body_ipos[self.torso] += rng.uniform(-1, 1, 3) * r["com_offset"] * lvl * np.array([1, 1, 0.3])
            for b in range(m.nbody):
                if b != self.torso and b > 0 and m.body_mass[b] < 1.0:
                    m.body_mass[b] *= 1 + rng.uniform(-1, 1) * r["link_mass_frac"] * lvl
            flo, fhi = r["friction"]
            fscale = rng.uniform(_lerp(1.0, flo, lvl), _lerp(1.0, fhi, lvl))
            m.geom_friction[:, 0] = self.nom_geom_friction[:, 0] * fscale
            kp_scale = 1 + rng.uniform(-1, 1, 8) * r["kp_frac"] * lvl
            m.actuator_gainprm[:, 0] = self.nom_gainprm[:, 0] * kp_scale
            m.actuator_biasprm[:, 1] = self.nom_biasprm[:, 1] * kp_scale
        self.torque_scale = 1 + self.np_random.uniform(-1, 1, 8) * r["torque_frac"] * lvl

    # ==================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "stage" in options:
            self.set_curriculum(options["stage"], options.get("progress", self.progress))
        c, m, d = self.cfg, self.model, self.data
        self.ep = p = self._episode_params()
        if "prone" in options:
            p["prone"] = bool(options["prone"])
        lvl = p["rand"]
        self._randomize_physics(lvl)
        lay = self.arena.sample(self.np_random, p["goal"], p["n_obstacles"], p["step_height"])
        self.goal = lay.goal
        mujoco.mj_resetData(m, d)

        # ---- 초기 자세 ----
        sz = c["robot"]["body_size"][2]
        g0 = self.arena.ground_height(*lay.start)
        q0 = np.zeros(8)
        if p["prone"]:
            z = g0 + sz / 2 + 0.003
        else:
            q0[1::2] = math.radians(90)
            z = g0 + c["robot"]["link2_length"] + c["robot"]["foot_radius"] - c["robot"]["hip_z"] + 0.003
        d.qpos[0:3] = [lay.start[0], lay.start[1], z]
        yaw = lay.start_yaw
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[self.jnt_qadr] = q0
        mujoco.mj_forward(m, d)

        # ---- 모터/통신/센서 내부 상태 ----
        rb = c["robot"]
        rng = self.np_random
        self.cmd = q0.copy()                       # 펌웨어가 믿는 관절각 (개루프)
        self.cmd_vel = np.zeros(8)
        self.slip = np.zeros(8)                    # 탈조로 인한 누적 오차
        self.backlash = np.radians(rng.uniform(*rb["backlash_deg"], 8))
        self.prev_action = np.zeros(8)
        self.applied_action = np.zeros(8)
        self.target = q0.copy()
        net = c["network"]
        extra = int(round(c["randomization"]["extra_latency_steps"] * lvl))
        self.act_latency = int(rng.integers(net["latency_steps"][0], net["latency_steps"][1] + extra + 1))
        self.act_queue = deque([np.zeros(8)] * (self.act_latency + 1), maxlen=self.act_latency + 1)
        s = c["sensors"]
        self.sens_latency = int(rng.integers(s["latency_steps"][0], s["latency_steps"][1] + extra + 1))
        self.gyro_bias = rng.normal(0, s["gyro_bias_std"], 3) * (0.5 + 0.5 * lvl)
        self.pos_err = np.zeros(2)
        first = self._measure_distances()
        self.dist_latest = first
        self.dist_queue = deque([first] * (self.sens_latency + 1), maxlen=self.sens_latency + 1)
        self.push_steps = 0

        # ---- 에피소드 통계 ----
        self.t = 0
        self.ep_sums = {k: 0.0 for k in REWARD_TERMS}
        self.ep_energy = 0.0
        self.stand_counter = 0
        self.stand_time = -1.0
        self.success = False
        self.reached = False
        self.air_time = np.zeros(4)
        self.last_contact = np.ones(4, bool)
        self.start_xy = d.qpos[0:2].copy()
        self.init_dist = self._goal_dist() if self.goal is not None else 0.0
        self.prev_dist = self.init_dist
        self.min_dist = self.init_dist
        self.trace = []

        frame = self._obs_frame()
        self.obs_hist = deque([frame] * self.hist, maxlen=self.hist)
        return np.concatenate(self.obs_hist).astype(np.float32), {"stage": p["stage"]}

    # ==================================================================
    def _goal_dist(self):
        return float(np.linalg.norm(self.goal - self.data.qpos[0:2]))

    def _measure_distances(self) -> np.ndarray:
        s = self.cfg["sensors"]
        d = self.data
        R = d.xmat[self.torso].reshape(3, 3)
        dmax = s["distance_max"]
        out = np.full(4, dmax)
        cone = math.radians(s["cone_half_angle_deg"])
        offsets = (-cone, 0.0, cone) if cone > 0 else (0.0,)
        pf = d.site_xpos[self.site_front]
        for i, yd in enumerate(s["distance_yaw_deg"]):
            best = dmax
            for off in offsets:
                yy = math.radians(yd) + off
                v = R @ np.array([math.cos(yy), math.sin(yy), 0.0])
                dist = mujoco.mj_ray(self.model, d, pf, v, self.ray_group, 1, -1, self._geomid)
                if 0 <= dist < best:
                    best = dist
            out[i] = best
        pd = d.site_xpos[self.site_down]
        pitch = math.radians(s["down_pitch_deg"])
        best = dmax
        for off in offsets:
            v = R @ np.array([math.cos(pitch) * math.cos(off), math.cos(pitch) * math.sin(off), -math.sin(pitch)])
            dist = mujoco.mj_ray(self.model, d, pd, v, self.ray_group, 1, -1, self._geomid)
            if 0 <= dist < best:
                best = dist
        out[3] = best
        return out

    def _noisy_distances(self, raw):
        s = self.cfg["sensors"]
        lvl = self.ep["rand"]
        out = raw + self.np_random.normal(0, s["noise_std"] * (0.5 + 0.5 * lvl), 4)
        drop = self.np_random.random(4) < s["dropout_prob"] * (0.5 + 0.5 * lvl)
        out[drop] = s["distance_max"]
        return np.clip(out, s["distance_min"], s["distance_max"])

    def _obs_frame(self) -> np.ndarray:
        c, d = self.cfg, self.data
        s = c["sensors"]
        lvl = self.ep["rand"]
        R = d.xmat[self.torso].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        grav = grav + self.np_random.normal(0, s["gravity_noise_std"], 3)
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3] + self.gyro_bias \
            + self.np_random.normal(0, s["gyro_noise_std"], 3)
        dists = self.dist_queue[0] / s["distance_max"]
        if self.goal is not None:
            est = d.qpos[0:2] + self.pos_err
            vec = self.goal - est
            dist = np.linalg.norm(vec)
            yaw = math.atan2(R[1, 0], R[0, 0])
            err = math.atan2(vec[1], vec[0]) - yaw
            goal = [min(dist, c["obs"]["goal_distance_clip"]) / c["obs"]["goal_distance_clip"],
                    math.cos(err), math.sin(err), 1.0]
        else:
            goal = [0.0, 1.0, 0.0, 0.0]
        parts = [grav, np.clip(gyro, -10, 10) * 0.25, self.cmd, self.prev_action, dists, goal]
        if c["obs"]["use_clock"]:
            ph = 2 * math.pi * c["obs"]["clock_hz"] * self.t * self.dt_ctrl
            parts.append([math.sin(ph), math.cos(ph)])
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in parts])

    # ==================================================================
    def step(self, action):
        c, m, d = self.cfg, self.model, self.data
        rb = c["robot"]
        p = self.ep
        w = p["weights"]
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)

        # ---- 통신: 패킷 손실 + 지연 ----
        if self.np_random.random() >= c["network"]["packet_loss"] * (0.5 + 0.5 * p["rand"]):
            self.act_queue.append(action.copy())
        else:
            self.act_queue.append(self.act_queue[-1].copy())
        applied = self.act_queue[0]
        sm = c["action"]["smoothing"]
        self.applied_action = sm * self.applied_action + (1 - sm) * applied
        self.target = np.clip(self.act_nominal + self.act_scale * self.applied_action, self.j_lo, self.j_hi)

        # ---- 펌웨어 가감속(속도 제한) ----
        # 사다리꼴 속도 프로파일: 최대 속도·가속도 제한, 목표 근처에서 감속
        vmax, amax, dt = rb["max_joint_speed"], rb["max_joint_accel"], self.dt_ctrl
        err = self.target - self.cmd
        v_des = np.sign(err) * np.minimum(vmax, np.sqrt(2 * amax * np.abs(err)))
        self.cmd_vel = self.cmd_vel + np.clip(v_des - self.cmd_vel, -amax * dt, amax * dt)
        cmd_old = self.cmd.copy()
        step = self.cmd_vel * dt
        # 목표를 지나치지 않도록
        over = np.abs(step) > np.abs(err)
        step[over] = err[over]
        self.cmd_vel[over] = 0.0
        self.cmd = self.cmd + step

        # ---- 외란 ----
        if self.push_steps > 0:
            self.push_steps -= 1
            if self.push_steps == 0:
                d.xfrc_applied[self.torso, :] = 0
        elif self.np_random.random() < c["randomization"]["push_prob"] * p["rand"]:
            ang = self.np_random.uniform(-math.pi, math.pi)
            f = c["randomization"]["push_force"] * self.np_random.uniform(0.3, 1.0)
            d.xfrc_applied[self.torso, 0:2] = [f * math.cos(ang), f * math.sin(ang)]
            self.push_steps = 3

        # ---- 물리 서브스텝 ----
        gear = rb["gear_ratio"]
        wmax = rb["motor_max_rpm"] * 2 * math.pi / 60
        kmin = rb["torque_at_max_speed"]
        half_bl = self.backlash / 2
        thr_extra = math.radians(rb["step_loss_threshold_deg"]
                                 + rb.get("step_loss_easy_margin_deg", 0.0) * (1.0 - p["rand"]))
        kp_now = m.actuator_gainprm[:, 0]
        tau_base = self.tau0 * self.torque_scale
        # 관절 qpos/qvel 주소가 연속이면 슬라이스(뷰)로 빠르게 접근
        qs, ds = self._qslice, self._dslice
        fr = m.actuator_forcerange
        step_loss = rb["step_loss"]
        slope = (1.0 - kmin) * gear / wmax
        delta = self.cmd - cmd_old
        e_pow = np.zeros(8)
        t_sq = np.zeros(8)
        inv_kp = 1.0 / kp_now
        for k in range(self.n_sub):
            motor = cmd_old + delta * ((k + 1) / self.n_sub) + self.slip
            q = d.qpos[qs]
            e = motor - q
            ae = np.abs(e)
            sgn = np.sign(e)
            d.ctrl[:] = np.where(ae <= half_bl, q, motor - sgn * half_bl)
            # 스테퍼 토크-속도 곡선 (선형 감소, 최고속 이상은 kmin 유지)
            tau = tau_base * np.maximum(1.0 - slope * np.abs(d.qvel[ds]), kmin)
            fr[:, 0] = -tau
            fr[:, 1] = tau
            mujoco.mj_step(m, d)
            if step_loss:
                # 토크 포화 상태에서 여유각(thr_extra)을 넘으면 탈조: 명령각과 실제각 사이 영구 오차
                over = ae - half_bl - tau * inv_kp - thr_extra
                if over.max() > 0:
                    lost = over > 0
                    self.slip[lost] -= sgn[lost] * over[lost]
            f = d.actuator_force
            e_pow += np.abs(f * d.qvel[ds])
            t_sq += f * f
        energy = float(e_pow.sum()) * self.dt_ctrl / self.n_sub
        torque_sq = float(t_sq.sum()) / self.n_sub
        self.ep_energy += energy
        self.t += 1

        # ---- 센서 갱신 (쿨타임 + 지연) ----
        s = c["sensors"]
        if self.t % s["update_period_steps"] == 0:
            self.dist_latest = self._noisy_distances(self._measure_distances())
        self.dist_queue.append(self.dist_latest)
        self.pos_err += self.np_random.normal(0, s["goal_noise_std"] * 0.1, 2)
        self.pos_err = np.clip(self.pos_err, -3 * s["goal_noise_std"], 3 * s["goal_noise_std"])

        # ---- 상태 ----
        pos = d.qpos[0:3]
        R = d.xmat[self.torso].reshape(3, 3)
        up_z = R[2, 2]
        height = pos[2] - self.arena.ground_height(pos[0], pos[1])
        contacts = np.zeros(4, bool)
        body_contact = 0
        obst_contact = 0
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = con.geom1, con.geom2
            for ga, gb in ((g1, g2), (g2, g1)):
                if ga in self.foot_lookup:
                    contacts[self.foot_lookup[ga]] = True
                    if gb in self.obstacle_geoms:
                        obst_contact += 0  # 발끝 접촉은 허용
                elif ga in self.robot_body_geoms:
                    if gb in self.obstacle_geoms:
                        obst_contact += 1
                    elif gb == self.floor_geom or gb in self.step_geoms:
                        body_contact += 1

        # ---- 보상 ----
        r = {}
        ht = w["stand_height"]
        h0 = c["robot"]["body_size"][2] / 2
        stand_frac = float(np.clip((height - h0) / (ht - h0), 0, 1))
        standing = stand_frac > 0.85 and up_z > 0.95
        r["height"] = w["height"] * stand_frac
        r["upright"] = w["upright"] * up_z
        r["alive"] = w["alive"]
        r["stand_bonus"] = w["stand_bonus"] * float(standing)
        r["body_contact"] = w["body_contact"] * float(body_contact > 0) * (1.0 if self.stand_time >= 0 or p["goal"] != "none" else 0.3)
        r["obstacle_contact"] = w["obstacle_contact"] * float(obst_contact > 0)
        r["energy"] = w["energy"] * energy / self.dt_ctrl
        r["torque"] = w["torque"] * torque_sq
        r["action_rate"] = w["action_rate"] * float(np.sum((action - self.prev_action) ** 2))
        q = d.qpos[self.jnt_qadr]
        mid, hr = (self.j_hi + self.j_lo) / 2, (self.j_hi - self.j_lo) / 2
        r["joint_limit"] = w["joint_limit"] * float(np.sum(np.clip(np.abs(q - mid) / hr - 0.92, 0, None) / 0.08))

        dist = 0.0
        if self.goal is not None:
            dist = self._goal_dist()
            r["progress"] = w["progress"] * (self.prev_dist - dist)
            self.prev_dist = dist
            self.min_dist = min(self.min_dist, dist)
            gvec = self.goal - pos[0:2]
            err = math.atan2(gvec[1], gvec[0]) - math.atan2(R[1, 0], R[0, 0])
            r["heading"] = w["heading"] * math.cos(err) * stand_frac
        else:
            r["progress"] = 0.0
            r["heading"] = 0.0

        # 발 들기 (착지 시 공중 시간 보상)
        self.air_time += self.dt_ctrl
        touchdown = contacts & ~self.last_contact
        moving = self.goal is not None and dist > c["arena"]["goal_radius"]
        r["feet_air_time"] = w["feet_air_time"] * float(np.sum((self.air_time - 0.15) * touchdown)) * float(moving)
        self.air_time[contacts] = 0.0
        self.last_contact = contacts

        # ---- 종료 판정 ----
        terminated = False
        fell = up_z < math.cos(math.radians(c["termination"]["flip_tilt_deg"]))
        out = c["termination"]["out_of_arena"] and (abs(pos[0]) > self.half + 0.2 or abs(pos[1]) > self.half + 0.2)
        r["reach"] = 0.0
        r["fall"] = 0.0
        if standing:
            self.stand_counter += 1
            if self.stand_time < 0 and self.stand_counter >= 10:
                self.stand_time = self.t * self.dt_ctrl
        else:
            self.stand_counter = 0
        if fell or out:
            terminated = True
            r["fall"] = w["fall"]
        if p["goal"] == "none":
            if self.stand_counter >= int(1.0 / self.dt_ctrl):
                self.success = True
        else:
            if dist < c["arena"]["goal_radius"]:
                self.reached = True
                self.success = True
                r["reach"] = w["reach"]
                terminated = True
            sp = p.get("success_progress")
            if sp is not None and (self.init_dist - dist) >= sp:
                self.success = True
        truncated = self.t >= p["max_steps"]

        reward = float(sum(r.values()))
        for k_, v_ in r.items():
            self.ep_sums[k_] += v_
        self.prev_action = action.copy()

        frame = self._obs_frame()
        self.obs_hist.append(frame)
        obs = np.concatenate(self.obs_hist).astype(np.float32)

        if self.record:
            self.trace.append({
                "t": self.t * self.dt_ctrl, "pos": pos.copy(), "yaw": math.atan2(R[1, 0], R[0, 0]),
                "up_z": up_z, "height": height, "q": q.copy(), "cmd": self.cmd.copy(),
                "target": self.target.copy(), "torque": d.actuator_force.copy(),
                "qvel": d.qvel[self.jnt_dadr].copy(), "contacts": contacts.copy(),
                "dists": self.dist_queue[0].copy(), "goal_dist": dist, "reward": dict(r),
                "energy": energy, "action": action.copy(),
            })

        info = {"stage": p["stage"]}
        if terminated or truncated:
            info["ep_stats"] = self._episode_stats(fell or out, truncated and not terminated, dist)
        return obs, reward, terminated, truncated, info

    def _episode_stats(self, fell, timeout, dist):
        p = self.ep
        T = self.t * self.dt_ctrl
        disp = float(np.linalg.norm(self.data.qpos[0:2] - self.start_xy))
        st = {
            "stage": p["stage"], "stage_name": p["name"], "progress": p["progress"], "rand_level": p["rand"],
            "n_obstacles": p["n_obstacles"], "step_height": p["step_height"], "prone_start": int(p["prone"]),
            "success": int(self.success), "reached": int(self.reached), "fell": int(fell), "timeout": int(timeout),
            "ep_len": self.t, "ep_time": T, "init_dist": self.init_dist, "final_dist": dist,
            "min_dist": self.min_dist, "goal_progress": self.init_dist - dist if self.goal is not None else 0.0,
            "displacement": disp, "mean_speed": (self.init_dist - dist) / max(T, 1e-6) if self.goal is not None else 0.0,
            "energy": self.ep_energy, "stand_time": self.stand_time,
            "act_latency": self.act_latency, "sens_latency": self.sens_latency,
            "slip_deg_mean": float(np.degrees(np.mean(np.abs(self.slip)))),
        }
        for k, v in self.ep_sums.items():
            st[f"r_{k}"] = v
        st["return"] = float(sum(self.ep_sums.values()))
        return st

    # ==================================================================
    def render(self, camera: str = "track", width: int = 640, height: int = 480):
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
