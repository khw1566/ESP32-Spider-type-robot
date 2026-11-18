"""경기장 배치: 출발점·목표 샘플링, 장애물(상자벽/드럼통/의자)·단차 배치.

재컴파일 없이 model.body_pos / geom_size 를 바꿔서 사용한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .model_builder import N_BOX_WALL, N_CHAIR, N_DRUM, N_STEP, PARK_POS


@dataclass
class Layout:
    start: np.ndarray
    start_yaw: float
    goal: np.ndarray | None
    obstacles: list = field(default_factory=list)   # (kind, x, y, yaw, size tuple, radius)
    steps: list = field(default_factory=list)       # (x, y, yaw, half_w, half_d, height)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class ArenaManager:
    def __init__(self, model: mujoco.MjModel, cfg: dict):
        self.m = model
        self.cfg = cfg
        self.half = cfg["sim"]["arena_size"] / 2
        a = cfg["arena"]
        self.a = a
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.wall_b = [bid(f"wall{i}") for i in range(N_BOX_WALL)]
        self.wall_g = [gid(f"wall{i}_g") for i in range(N_BOX_WALL)]
        self.drum_b = [bid(f"drum{i}") for i in range(N_DRUM)]
        self.drum_g = [gid(f"drum{i}_g") for i in range(N_DRUM)]
        self.chair_b = [bid(f"chair{i}") for i in range(N_CHAIR)]
        self.step_b = [bid(f"step{i}") for i in range(N_STEP)]
        self.step_g = [gid(f"step{i}_g") for i in range(N_STEP)]
        self.goal_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.goal_pole = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_pole")
        self.layout: Layout | None = None

    # ------------------------------------------------------------------
    def _park_all(self):
        px, py = PARK_POS
        for k, b in enumerate(self.wall_b + self.drum_b + self.chair_b + self.step_b):
            self.m.body_pos[b] = [px + 2 * k, py, 0.0]

    def _set_box(self, b, g, x, y, yaw, hx, hy, hz, z0=0.0):
        self.m.body_pos[b] = [x, y, z0]
        self.m.body_quat[b] = _yaw_quat(yaw)
        self.m.geom_size[g] = [hx, hy, hz]
        self.m.geom_pos[g] = [0, 0, hz]
        self.m.geom_rbound[g] = math.sqrt(hx * hx + hy * hy + hz * hz)

    def _set_drum(self, b, g, x, y, r, h):
        self.m.body_pos[b] = [x, y, 0.0]
        self.m.geom_size[g] = [r, h / 2, 0]
        self.m.geom_pos[g] = [0, 0, h / 2]
        self.m.geom_rbound[g] = math.sqrt(r * r + (h / 2) ** 2)

    def _inside(self, p, margin):
        return abs(p[0]) <= self.half - margin and abs(p[1]) <= self.half - margin

    # ------------------------------------------------------------------
    def sample(self, rng: np.random.Generator, goal_cfg, n_obstacles: int, step_height: float) -> Layout:
        """새 배치를 샘플링하고 모델에 반영한다."""
        self._park_all()
        margin = self.a["margin"]
        goal = None
        for _ in range(200):
            if goal_cfg == "none" or goal_cfg is None:
                start = rng.uniform(-1.0, 1.0, size=2)
                yaw = rng.uniform(-math.pi, math.pi)
                break
            dmin, dmax = goal_cfg["distance"]
            dist = rng.uniform(dmin, dmax)
            start = rng.uniform(-(self.half - margin), self.half - margin, size=2)
            if goal_cfg["mode"] == "forward":
                # 정면으로 dist 만큼: 벽까지 여유 있는 출발점·방향 선택
                yaw = rng.uniform(-math.pi, math.pi)
                g = start + dist * np.array([math.cos(yaw), math.sin(yaw)])
            else:
                ang = rng.uniform(-math.pi, math.pi)
                g = start + dist * np.array([math.cos(ang), math.sin(ang)])
                yaw = rng.uniform(-math.pi, math.pi)
            if self._inside(g, margin):
                goal = g
                break
        else:
            start, yaw, goal = np.zeros(2), 0.0, np.array([1.0, 0.0])
        lay = Layout(start=np.asarray(start, float), start_yaw=float(yaw), goal=goal)

        clear = self.a["obstacle_clearance"]

        def free(p, r):
            if not self._inside(p, r * 0.5):
                return False
            if np.linalg.norm(p - lay.start) < clear + r:
                return False
            if goal is not None and np.linalg.norm(p - goal) < clear + r:
                return False
            for o in lay.obstacles:
                if np.linalg.norm(p - np.array(o[1:3])) < o[5] + r + 0.05:
                    return False
            return True

        def sample_point(r):
            for _ in range(60):
                # 출발-목표 경로 근처에 우선 배치 (회피가 필요하도록)
                if goal is not None and rng.random() < 0.7:
                    t = rng.uniform(0.25, 0.75)
                    p = lay.start + t * (goal - lay.start) + rng.normal(0, 0.35, 2)
                else:
                    p = rng.uniform(-self.half, self.half, 2)
                if free(p, r):
                    return p
            return None

        # ---- 장애물 ----
        kinds = ["wall"] * N_BOX_WALL + ["drum"] * N_DRUM + ["chair"] * N_CHAIR
        rng.shuffle(kinds)
        used = {"wall": 0, "drum": 0, "chair": 0}
        for kind in kinds:
            if len(lay.obstacles) >= n_obstacles:
                break
            if kind == "wall" and used["wall"] < N_BOX_WALL:
                c = self.a["box_wall"]
                L, T, H = (rng.uniform(*c["length"]), rng.uniform(*c["thickness"]), rng.uniform(*c["height"]))
                r = math.hypot(L, T) / 2
                p = sample_point(r)
                if p is None:
                    continue
                yaw_o = rng.uniform(-math.pi, math.pi)
                i = used["wall"]
                self._set_box(self.wall_b[i], self.wall_g[i], p[0], p[1], yaw_o, L / 2, T / 2, H / 2)
                lay.obstacles.append(("wall", p[0], p[1], yaw_o, (L, T, H), r))
                used["wall"] += 1
            elif kind == "drum" and used["drum"] < N_DRUM:
                c = self.a["drum"]
                R, H = rng.uniform(*c["radius"]), rng.uniform(*c["height"])
                p = sample_point(R)
                if p is None:
                    continue
                i = used["drum"]
                self._set_drum(self.drum_b[i], self.drum_g[i], p[0], p[1], R, H)
                lay.obstacles.append(("drum", p[0], p[1], 0.0, (R, H), R))
                used["drum"] += 1
            elif kind == "chair" and used["chair"] < N_CHAIR:
                sw, sd = self.a["chair"]["seat"]
                r = math.hypot(sw, sd) / 2
                p = sample_point(r)
                if p is None:
                    continue
                yaw_o = rng.uniform(-math.pi, math.pi)
                i = used["chair"]
                self.m.body_pos[self.chair_b[i]] = [p[0], p[1], 0.0]
                self.m.body_quat[self.chair_b[i]] = _yaw_quat(yaw_o)
                lay.obstacles.append(("chair", p[0], p[1], yaw_o, (sw, sd), r))
                used["chair"] += 1

        # ---- 단차: 출발-목표 사이를 가로지르는 단상 ----
        if step_height > 0.0 and goal is not None:
            c = self.a["step"]
            d = goal - lay.start
            dist = np.linalg.norm(d)
            heading = math.atan2(d[1], d[0])
            n_steps = 1 if rng.random() < 0.6 or dist < 2.0 else 2
            for i in range(min(n_steps, N_STEP)):
                w, dep = rng.uniform(*c["width"]), rng.uniform(*c["depth"])
                # 첫 번째 단차는 경로 중간, 두 번째는 목표가 단 위에 올라가도록
                t = rng.uniform(0.35, 0.6) if i == 0 else 1.0
                center = lay.start + t * d
                if i == 0:
                    dep = min(dep, max(0.3, dist * 0.35))
                self._set_box(self.step_b[i], self.step_g[i], center[0], center[1], heading,
                              dep / 2, w / 2, step_height / 2)
                lay.steps.append((center[0], center[1], heading, w / 2, dep / 2, step_height))

        # ---- 목표 표시 ----
        if goal is not None:
            gz = self.ground_height(goal[0], goal[1], lay)
            self.m.site_pos[self.goal_site] = [goal[0], goal[1], gz + 0.002]
            self.m.site_pos[self.goal_pole] = [goal[0], goal[1], gz + 0.2]
        else:
            self.m.site_pos[self.goal_site] = [PARK_POS[0], PARK_POS[1], -1]
            self.m.site_pos[self.goal_pole] = [PARK_POS[0], PARK_POS[1], -1]
        self.layout = lay
        return lay

    def ground_height(self, x: float, y: float, lay: Layout | None = None) -> float:
        lay = lay or self.layout
        h = 0.0
        if lay is None:
            return h
        for (cx, cy, yaw, hw, hd, sh) in lay.steps:
            dx, dy = x - cx, y - cy
            c, s = math.cos(-yaw), math.sin(-yaw)
            lx, ly = c * dx - s * dy, s * dx + c * dy
            if abs(lx) <= hd and abs(ly) <= hw:
                h = max(h, sh)
        return h
