"""학습된 정책 평가 + 로봇 거동 그래프(보행 패턴, 관절 궤적, 토크, 궤적 등) + 영상.

사용법
  python scripts/evaluate.py runs/<실행>/checkpoints/latest.zip --stage 3 --episodes 20
  python scripts/evaluate.py runs/<실행>/checkpoints/latest.zip --stage 7 --episodes 50 --video
  python scripts/evaluate.py --policy zero --stage 0      # 비교 기준선 (zero / random / stand)

출력 (기본: <체크포인트 폴더>/../eval_<태그>_stage<k>/)
  summary.json, episodes.csv, ep000_trace.npz, 그래프 PNG, (선택) mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from spider_rl.config import load_config  # noqa: E402
from spider_rl.env import REWARD_TERMS, SpiderEnv  # noqa: E402
from spider_rl.model_builder import JOINT_NAMES, LEG_NAMES, output_torque  # noqa: E402
from spider_rl.plotting import PALETTE, TEXT2, L, savefig, setup_style  # noqa: E402


def load_policy(path, kind, env):
    if kind == "zero":
        return lambda o: np.zeros(8), None
    if kind == "random":
        return lambda o: env.action_space.sample(), None
    if kind == "stand":   # 천천히 무릎을 내려 일어서는 스크립트 정책 (기준선)
        st = {"i": 0}

        def f(o):
            st["i"] += 1
            a = np.zeros(8)
            a[1::2] = -0.6 + 1.2 * min(1.0, st["i"] / 30)
            return a
        return f, st
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model = PPO.load(path, device="cpu")
    vn_path = path.replace(".zip", "_vecnormalize.pkl")
    vn = None
    if os.path.exists(vn_path):
        vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: SpiderEnv(env.cfg)]))
        vn.training = False
        vn.norm_reward = False

    def f(o):
        oo = vn.normalize_obs(o[None]) if vn is not None else o[None]
        a, _ = model.predict(oo, deterministic=True)
        return a[0]
    return f, None


def plot_episode(tr, env, out, tag, saved):
    t = np.array([s["t"] for s in tr])
    q = np.degrees(np.array([s["q"] for s in tr]))
    cmd = np.degrees(np.array([s["cmd"] for s in tr]))
    tau = np.array([s["torque"] for s in tr])
    qv = np.array([s["qvel"] for s in tr])
    con = np.array([s["contacts"] for s in tr])
    pos = np.array([s["pos"] for s in tr])
    dists = np.array([s["dists"] for s in tr])
    rb = env.cfg["robot"]

    # 1) 보행 패턴 (발 접지 다이어그램)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for i, leg in enumerate(LEG_NAMES):
        on = con[:, i]
        starts = np.where(np.diff(np.r_[0, on.astype(int)]) == 1)[0]
        ends = np.where(np.diff(np.r_[on.astype(int), 0]) == -1)[0]
        for s0, e0 in zip(starts, ends):
            ax.broken_barh([(t[s0], t[e0] - t[s0] + env.dt_ctrl)], (i - 0.35, 0.7), color=PALETTE[0])
    ax.set_yticks(range(4))
    ax.set_yticklabels(LEG_NAMES)
    ax.set_xlabel(L("시간 (s)", "time (s)"))
    ax.set_title(L("보행 패턴: 발 접지 다이어그램 (막대=접지)", "Gait diagram (bar = stance)"))
    ax.grid(False)
    savefig(fig, out, f"{tag}_gait_diagram", saved)

    # 2) 관절 궤적: 명령각 vs 실제각 (유격·탈조 오차 확인)
    fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j // 2, j % 2]
        ax.plot(t, cmd[:, j], color=PALETTE[0], label=L("명령(개루프)", "command"))
        ax.plot(t, q[:, j], color=PALETTE[1], label=L("실제", "actual"))
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(L("시간 (s)", "time (s)"))
    fig.suptitle(L("관절 궤적 (deg): 명령 vs 실제", "Joint trajectories (deg)"))
    savefig(fig, out, f"{tag}_joint_trajectories", saved)

    # 3) 토크-속도 운용점 vs 모터 곡선
    fig, ax = plt.subplots(figsize=(6, 4))
    w_motor = np.abs(qv) * rb["gear_ratio"] * 60 / (2 * math.pi)
    ax.scatter(w_motor.ravel(), np.abs(tau).ravel(), s=3, alpha=0.3, color=PALETTE[0], label=L("운용점", "operating points"))
    rpm = np.linspace(0, rb["motor_max_rpm"] * 1.1, 100)
    t0 = output_torque(rb)
    curve = t0 * np.maximum(1 - (1 - rb["torque_at_max_speed"]) * rpm / rb["motor_max_rpm"], rb["torque_at_max_speed"])
    ax.plot(rpm, curve, color=PALETTE[1], label=L("최대 토크 곡선", "torque limit"))
    ax.set_xlabel(L("모터 속도 (rpm)", "motor speed (rpm)"))
    ax.set_ylabel(L("관절 토크 |τ| (N·m)", "|joint torque| (N·m)"))
    ax.set_title(L("토크-속도 운용 영역", "Torque-speed operating region"))
    ax.legend(fontsize=8)
    savefig(fig, out, f"{tag}_torque_speed", saved)

    # 4) 관절 토크 시계열 + 누적 에너지 (별도 패널)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for j in range(1, 8, 2):
        axes[0].plot(t, tau[:, j], color=PALETTE[j // 2], label=JOINT_NAMES[j])
    axes[0].axhline(t0, color=TEXT2, ls="--", lw=0.8)
    axes[0].axhline(-t0, color=TEXT2, ls="--", lw=0.8)
    axes[0].set_ylabel("N·m")
    axes[0].set_title(L("무릎 토크 (점선=정지 토크 한계)", "Knee torque (dashed = limit)"))
    axes[0].legend(fontsize=7, ncol=4)
    axes[1].plot(t, np.cumsum([s["energy"] for s in tr]), color=PALETTE[0])
    axes[1].set_ylabel("J")
    axes[1].set_xlabel(L("시간 (s)", "time (s)"))
    axes[1].set_title(L("누적 기계적 에너지", "Cumulative mechanical energy"))
    savefig(fig, out, f"{tag}_torque_energy", saved)

    # 5) 몸체 높이·기울기
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    axes[0].plot(t, [s["height"] * 100 for s in tr], color=PALETTE[0])
    axes[0].axhline(env.cfg["reward"]["stand_height"] * 100, color=TEXT2, ls="--", lw=0.8)
    axes[0].set_ylabel("cm")
    axes[0].set_title(L("몸체 높이 (점선=서기 목표)", "Body height"))
    axes[1].plot(t, [math.degrees(math.acos(np.clip(s["up_z"], -1, 1))) for s in tr], color=PALETTE[1])
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel(L("시간 (s)", "time (s)"))
    axes[1].set_title(L("몸체 기울기", "Body tilt"))
    savefig(fig, out, f"{tag}_height_tilt", saved)

    # 6) 거리센서 판독값
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, lab in enumerate([L("전방 0°", "front"), L("좌 45°", "left 45"), L("우 45°", "right 45"), L("하향", "down")]):
        ax.plot(t, dists[:, i] * 100, color=PALETTE[i], label=lab)
    ax.set_ylabel("cm")
    ax.set_xlabel(L("시간 (s)", "time (s)"))
    ax.set_title(L("거리센서 관측값 (지연·쿨타임·노이즈 포함)", "Distance sensors (as observed)"))
    ax.legend(fontsize=7, ncol=4)
    savefig(fig, out, f"{tag}_distance_sensors", saved)

    # 7) 보상 항목 누적
    fig, ax = plt.subplots(figsize=(8, 3.8))
    terms = [k for k in REWARD_TERMS if any(abs(s["reward"].get(k, 0)) > 1e-9 for s in tr)]
    mags = sorted(terms, key=lambda k: -abs(sum(s["reward"].get(k, 0) for s in tr)))[:8]
    for i, k in enumerate(mags):
        ax.plot(t, np.cumsum([s["reward"].get(k, 0) for s in tr]), color=PALETTE[i], label=k)
    ax.set_xlabel(L("시간 (s)", "time (s)"))
    ax.set_title(L("보상 항목별 누적 (상위 8개)", "Cumulative reward terms (top 8)"))
    ax.legend(fontsize=7, ncol=4)
    savefig(fig, out, f"{tag}_reward_cumulative", saved)
    return pos


def _draw_layout(ax, env, lay, pos, ok):
    h = env.half
    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, ec=TEXT2, lw=1.2))
    rot = lambda yaw: np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    for (cx, cy, yaw, hw, hd, sh) in lay.steps:
        pts = (rot(yaw) @ np.array([[hd, hw], [-hd, hw], [-hd, -hw], [hd, -hw]]).T).T + [cx, cy]
        ax.add_patch(Polygon(pts, closed=True, fc="#e3e1da", ec="#b5b3ab", lw=0.6))
        ax.text(cx, cy, f"{sh * 100:.0f}cm", fontsize=6, color=TEXT2, ha="center", va="center")
    for kind, x, y, yaw, size, r in lay.obstacles:
        if kind == "drum":
            ax.add_patch(Circle((x, y), size[0], fc="#8f8d86", ec="none"))
        else:
            hx, hy = size[0] / 2, size[1] / 2
            pts = (rot(yaw) @ np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]).T).T + [x, y]
            ax.add_patch(Polygon(pts, closed=True, fc="#8f8d86", ec="none"))
    ax.plot(pos[:, 0], pos[:, 1], color=PALETTE[0], lw=1.6)
    ax.plot(pos[0, 0], pos[0, 1], "o", color=PALETTE[0], ms=5)
    if lay.goal is not None:
        ax.add_patch(Circle(lay.goal, env.cfg["arena"]["goal_radius"], fc="none", ec=PALETTE[1], lw=1.6))
    ax.set_xlim(-h - 0.1, h + 0.1)
    ax.set_ylim(-h - 0.1, h + 0.1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_title(L("성공", "success") if ok else L("실패", "fail"), fontsize=9)


def plot_layout(env, trajs, out, saved, name="trajectories_xy"):
    """경기장·장애물(진회색)·단차(연회색)·목표(원)와 함께 에피소드별 이동 궤적 (최대 4개)."""
    n = min(4, len(trajs))
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for ax, (lay, pos, ok) in zip(axes.flat, trajs[:n]):
        _draw_layout(ax, env, lay, pos, ok)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(L("이동 궤적 (●출발, ○목표, 진회색=장애물, 연회색=단차)",
                   "Trajectories (dot=start, circle=goal, dark=obstacle, light=step)"), fontsize=10)
    savefig(fig, out, name, saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--policy", default="model", choices=["model", "zero", "random", "stand"])
    ap.add_argument("--config", default=None, help="기본: 체크포인트 실행 폴더의 config.yaml")
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--progress", type=float, default=1.0, help="단계 내 난이도 (0~1)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--plot-episodes", type=int, default=3)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--camera", default="track")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_style()

    cfg_path = args.config
    if cfg_path is None and args.checkpoint:
        cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))), "config.yaml")
        cfg_path = cand if os.path.exists(cand) else None
    cfg = load_config(cfg_path)
    stage = args.stage
    if stage is None and args.checkpoint:
        cj = args.checkpoint.replace(".zip", "_curriculum.json")
        stage = json.load(open(cj))["stage"] if os.path.exists(cj) else 0
    stage = stage or 0
    env = SpiderEnv(cfg, stage=stage, progress=args.progress, record=True, fixed_stage=True)
    policy, st = load_policy(args.checkpoint, args.policy if args.checkpoint or args.policy != "model" else "zero", env)
    tag = args.policy if args.policy != "model" else os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = args.out or (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))),
                                    f"eval_{tag}_stage{stage}") if args.checkpoint else f"eval_{tag}_stage{stage}")
    os.makedirs(out, exist_ok=True)
    saved: list[str] = []
    writer = None
    if args.video:
        try:
            import imageio
            writer = imageio.get_writer(os.path.join(out, f"{tag}_stage{stage}.mp4"), fps=30)
        except Exception as ex:  # noqa: BLE001
            print("영상 저장 불가 (pip install imageio imageio-ffmpeg):", ex)

    rows, trajs = [], []
    for ep in range(args.episodes):
        if st is not None:
            st["i"] = 0
        obs, _ = env.reset(seed=args.seed + ep)
        lay = env.arena.layout
        done = False
        while not done:
            a = policy(obs)
            obs, r, te, tr, info = env.step(a)
            done = te or tr
            if writer is not None and ep < 3:
                writer.append_data(env.render(camera=args.camera))
        s = info["ep_stats"]
        rows.append(s)
        pos = np.array([x["pos"] for x in env.trace])
        trajs.append((lay, pos, bool(s["success"])))
        if ep < args.plot_episodes:
            plot_episode(env.trace, env, out, f"ep{ep:03d}", saved)
            np.savez_compressed(os.path.join(out, f"ep{ep:03d}_trace.npz"),
                                **{k: np.array([x[k] for x in env.trace]) for k in
                                   ["t", "pos", "yaw", "up_z", "height", "q", "cmd", "target", "torque", "qvel",
                                    "contacts", "dists", "goal_dist", "energy", "action"]})
        print(f"ep {ep:3d}  success={s['success']} fell={s['fell']} len={s['ep_len']} "
              f"dist {s['init_dist']:.2f}->{s['final_dist']:.2f}  return={s['return']:.1f}")
    if writer is not None:
        writer.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "episodes.csv"), index=False)
    if any(lay.goal is not None for lay, _, _ in trajs):
        plot_layout(env, trajs, out, saved)
    summary = {"checkpoint": args.checkpoint, "policy": args.policy, "stage": stage,
               "stage_name": cfg["curriculum"]["stages"][stage]["name"], "episodes": len(df),
               "success_rate": float(df["success"].mean()), "fall_rate": float(df["fell"].mean()),
               "timeout_rate": float(df["timeout"].mean()), "mean_return": float(df["return"].mean()),
               "mean_final_dist": float(df["final_dist"].mean()), "mean_ep_time": float(df["ep_time"].mean()),
               "mean_stand_time": float(df.loc[df["stand_time"] >= 0, "stand_time"].mean())
               if (df["stand_time"] >= 0).any() else None}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[그래프] {len(saved)}개 → {out}")


if __name__ == "__main__":
    main()
