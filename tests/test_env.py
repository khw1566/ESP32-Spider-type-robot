"""기본 동작 테스트: python -m pytest -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spider_rl.config import load_config  # noqa: E402
from spider_rl.env import SpiderEnv  # noqa: E402
from spider_rl.model_builder import robot_mass_budget  # noqa: E402


def test_mass_budget_matches_spec():
    total = robot_mass_budget(load_config())["total"]
    assert 4.0 <= total <= 4.6   # 사양: 약 4~4.5 kg


def test_check_env():
    from stable_baselines3.common.env_checker import check_env
    check_env(SpiderEnv(stage=0), warn=False)


@pytest.mark.parametrize("stage", range(8))
def test_random_rollout_all_stages(stage):
    env = SpiderEnv(stage=stage, progress=1.0, fixed_stage=True)
    obs, _ = env.reset(seed=stage)
    assert obs.shape == env.observation_space.shape
    for _ in range(150):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if te or tr:
            assert "ep_stats" in info
            obs, _ = env.reset()


def test_scripted_stand_up_is_feasible():
    """1초에 걸쳐 무릎을 내리면 탈조 없이 일어설 수 있어야 한다 (토크 충분성 검증)."""
    env = SpiderEnv(stage=0, fixed_stage=True)
    env.reset(seed=1, options={"prone": True})
    for i in range(60):
        a = np.zeros(8)
        a[1::2] = -0.6 + 1.2 * min(1.0, i / 30)
        env.step(a)
    assert env.stand_time > 0
    assert np.degrees(np.abs(env.slip).max()) < 1.0


def test_obstacles_and_steps_placed():
    env = SpiderEnv(stage=7, progress=1.0, fixed_stage=True)
    env.reset(seed=3)
    lay = env.arena.layout
    assert lay.goal is not None
    d = np.linalg.norm(lay.goal - lay.start)
    assert 1.0 - 1e-6 <= d <= 4.0 + 1e-6
    assert len(lay.obstacles) >= 1 and len(lay.steps) >= 1


def test_distance_sensor_sees_obstacle():
    env = SpiderEnv(stage=0, fixed_stage=True)
    env.reset(seed=0, options={"prone": True})
    base = env._measure_distances()
    assert base[0] == pytest.approx(env.cfg["sensors"]["distance_max"]) or base[0] > 0.5
    # 로봇 정면 0.5 m에 벽 배치
    import math
    yaw = env.arena.layout.start_yaw
    p = env.data.qpos[0:2] + 0.6 * np.array([math.cos(yaw), math.sin(yaw)])
    env.arena._set_box(env.arena.wall_b[0], env.arena.wall_g[0], p[0], p[1], yaw + math.pi / 2, 0.5, 0.05, 0.2)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    d = env._measure_distances()
    assert 0.3 < d[0] < 0.6


def test_dashboard_control_roundtrip(tmp_path):
    from spider_rl import dashboard_backend as be
    d = tmp_path / "run"
    d.mkdir()
    be.write_control(str(d), max_hours=7.5, stop=False)
    be.write_control(str(d), max_hours=9.0)
    ctrl = be.read_control(str(d))
    assert ctrl["max_hours"] == 9.0 and ctrl["stop"] is False
    cmd = be.build_train_command({"run_name": "x", "preset": "desktop", "n_envs": 12, "max_hours": 7.5,
                                  "timesteps": 1e6, "stage": "3", "fixed_stage": True,
                                  "sets": ["ppo.ent_coef=0.005", "# 주석", ""]}, runs_dir=str(tmp_path))
    assert "--max-hours" in cmd and cmd[cmd.index("--max-hours") + 1] == "7.5"
    assert cmd.count("--set") == 1 and "--fixed-stage" in cmd and str(tmp_path) in cmd
