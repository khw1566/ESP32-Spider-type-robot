"""에피소드 녹화·재생·재현.

녹화 파일(.npz) 하나에 3D 재생과 재현에 필요한 정보를 모두 담는다.
  - qpos (T+1, nq)      : 매 제어 스텝(30 Hz)의 전체 자세 → 3D 재생은 물리 계산 없이 정확히 같은 장면
  - 경기장 배치          : 장애물·단차·목표 위치/크기 (model 의 가변 필드 스냅샷)
  - 센서·관절·토크·접지·보상 시계열 : 화면 표시와 그래프용
  - 재현 정보            : 설정(config), 시드, 단계, 진행도, 정책(체크포인트 경로) → 같은 조건으로 다시 시뮬레이션
"""
from __future__ import annotations

import json
import math
import os
import time

import mujoco
import numpy as np
import yaml

from .env import SpiderEnv

LAYOUT_FIELDS = [("body_pos", "body"), ("body_quat", "body"), ("geom_size", "geom"), ("geom_pos", "geom"),
                 ("geom_rbound", "geom"), ("site_pos", "site")]
SERIES = ["t", "pos", "yaw", "up_z", "height", "q", "cmd", "target", "torque", "qvel", "contacts", "dists",
          "goal_dist", "energy", "action"]


# ----------------------------------------------------------------------
# 정책 로딩 (평가·뷰어 공용)
# ----------------------------------------------------------------------
def load_model_policy(checkpoint: str, cfg: dict):
    """체크포인트(.zip) + VecNormalize 통계 → obs -> action 함수 (결정적)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model = PPO.load(checkpoint, device="cpu")
    vn = None
    vn_path = checkpoint.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vn_path):
        vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: SpiderEnv(cfg)]))
        vn.training = False
        vn.norm_reward = False

    def policy(obs):
        o = vn.normalize_obs(obs[None]) if vn is not None else obs[None]
        a, _ = model.predict(o, deterministic=True)
        return a[0]
    return policy


def scripted_policy(kind: str, env: SpiderEnv):
    """비교용 기준선: zero(가만히) / random / stand(1초에 걸쳐 일어서기)."""
    state = {"i": 0}

    def policy(obs):
        state["i"] += 1
        if kind == "random":
            return env.action_space.sample()
        a = np.zeros(8)
        if kind == "stand":
            a[1::2] = -0.6 + 1.2 * min(1.0, state["i"] / 30)
        return a
    policy.reset = lambda: state.update(i=0)
    return policy


def run_dir_of(checkpoint: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))


def config_for(checkpoint: str | None):
    from .config import load_config
    if checkpoint:
        cand = os.path.join(run_dir_of(checkpoint), "config.yaml")
        if os.path.exists(cand):
            return load_config(cand)
    return load_config()


# ----------------------------------------------------------------------
# 녹화
# ----------------------------------------------------------------------
def snapshot_layout(model: mujoco.MjModel) -> dict:
    return {name: getattr(model, name).copy() for name, _ in LAYOUT_FIELDS}


def apply_layout(model: mujoco.MjModel, layout: dict) -> None:
    for name, _ in LAYOUT_FIELDS:
        getattr(model, name)[:] = layout[name]


def record_episode(env: SpiderEnv, policy, seed: int, stage: int, progress: float = 1.0,
                   prone: bool | None = None, meta: dict | None = None, on_step=None) -> dict:
    """정책으로 한 에피소드를 돌리며 녹화. env 는 record=True, fixed_stage=True 권장."""
    env.record = True
    env.fixed_stage = True
    env.set_curriculum(stage, progress)
    if hasattr(policy, "reset"):
        policy.reset()
    opts = {"stage": stage, "progress": progress}
    if prone is not None:
        opts["prone"] = prone
    obs, _ = env.reset(seed=seed, options=opts)
    env.action_space.seed(seed)   # random 기준선도 재현되도록
    q0 = env.data.qpos.copy()
    layout = snapshot_layout(env.model)
    lay = env.arena.layout
    done = False
    info = {}
    while not done:
        a = policy(obs)
        obs, r, te, tr, info = env.step(a)
        done = te or tr
        if on_step is not None and on_step(env) is False:
            break
    tr_ = env.trace
    rec = {
        "qpos": np.vstack([q0] + [s["qpos"] for s in tr_]),
        "layout": layout,
        "series": {k: np.array([s[k] for s in tr_]) for k in SERIES},
        "rewards": {k: np.array([s["reward"].get(k, 0.0) for s in tr_]) for k in tr_[0]["reward"]} if tr_ else {},
        "stats": info.get("ep_stats", {}),
        "goal": None if lay.goal is None else np.asarray(lay.goal, float),
        "start": np.asarray(lay.start, float),
        "meta": dict(meta or {}, seed=int(seed), stage=int(stage), progress=float(progress), prone=prone,
                     dt=env.dt_ctrl, recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                     stage_name=env.stages[stage]["name"]),
        "config": env.cfg,
    }
    return rec


def save_replay(rec: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {"qpos": rec["qpos"], "start": rec["start"]}
    if rec["goal"] is not None:
        arrays["goal"] = rec["goal"]
    for k, v in rec["layout"].items():
        arrays[f"layout__{k}"] = v
    for k, v in rec["series"].items():
        arrays[f"series__{k}"] = v
    for k, v in rec["rewards"].items():
        arrays[f"reward__{k}"] = v
    stats = {k: (v.item() if hasattr(v, "item") else v) for k, v in rec["stats"].items()}
    arrays["meta_json"] = np.array(json.dumps({"meta": rec["meta"], "stats": stats}, ensure_ascii=False))
    arrays["config_yaml"] = np.array(yaml.safe_dump(rec["config"], allow_unicode=True, sort_keys=False))
    np.savez_compressed(path, **arrays)
    return path


def load_replay(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    mj = json.loads(str(z["meta_json"]))
    rec = {"qpos": z["qpos"], "start": z["start"], "goal": z["goal"] if "goal" in z else None,
           "layout": {}, "series": {}, "rewards": {}, "meta": mj["meta"], "stats": mj["stats"],
           "config": yaml.safe_load(str(z["config_yaml"])), "path": path}
    for k in z.files:
        if k.startswith("layout__"):
            rec["layout"][k[8:]] = z[k]
        elif k.startswith("series__"):
            rec["series"][k[8:]] = z[k]
        elif k.startswith("reward__"):
            rec["rewards"][k[8:]] = z[k]
    return rec


def replay_name(stage: int, seed: int, tag: str = "") -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_stage{stage}_seed{seed}" + (f"_{tag}" if tag else "") + ".npz"


def list_replays(run_dir: str) -> list[str]:
    d = os.path.join(run_dir, "replays")
    if not os.path.isdir(d):
        return []
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
    return sorted(files, key=os.path.getmtime, reverse=True)


# ----------------------------------------------------------------------
# 재현
# ----------------------------------------------------------------------
def reproduce(rec: dict, policy=None, checkpoint: str | None = None) -> dict:
    """녹화 파일의 설정·시드·단계·정책으로 다시 시뮬레이션하고 원본과 비교."""
    m = rec["meta"]
    env = SpiderEnv(rec["config"], stage=m["stage"], progress=m["progress"], record=True, fixed_stage=True)
    if policy is None:
        ck = checkpoint or m.get("checkpoint")
        if ck and os.path.exists(ck):
            policy = load_model_policy(ck, rec["config"])
        elif m.get("policy") in ("zero", "random", "stand"):
            policy = scripted_policy(m["policy"], env)
        else:
            raise FileNotFoundError(f"정책 체크포인트를 찾을 수 없습니다: {ck}")
    new = record_episode(env, policy, m["seed"], m["stage"], m["progress"], m.get("prone"), meta=m)
    T = min(len(new["qpos"]), len(rec["qpos"]))
    err = float(np.max(np.abs(new["qpos"][:T] - rec["qpos"][:T]))) if T else float("nan")
    return {"rec": new, "max_qpos_error": err, "same_length": len(new["qpos"]) == len(rec["qpos"]),
            "orig_success": rec["stats"].get("success"), "new_success": new["stats"].get("success")}


# ----------------------------------------------------------------------
# 표시 보조: 거리센서 광선
# ----------------------------------------------------------------------
def sensor_segments(model, data, cfg, dists):
    """현재 자세에서 거리센서 4개(각 중심 광선)의 시작·끝점. dists 는 m 단위 측정값."""
    s = cfg["sensors"]
    torso = model.body("torso").id
    R = data.xmat[torso].reshape(3, 3)
    pf = data.site_xpos[model.site("dist_front").id].copy()
    pd = data.site_xpos[model.site("dist_down").id].copy()
    segs = []
    for i, yd in enumerate(s["distance_yaw_deg"]):
        y = math.radians(yd)
        v = R @ np.array([math.cos(y), math.sin(y), 0.0])
        segs.append((pf, pf + v * float(dists[i])))
    p = math.radians(s["down_pitch_deg"])
    v = R @ np.array([math.cos(p), 0.0, -math.sin(p)])
    segs.append((pd, pd + v * float(dists[3])))
    return segs
