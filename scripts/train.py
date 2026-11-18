"""PPO 학습 실행 스크립트.

예시
  # 처음부터 (일어서기 단계부터 커리큘럼 자동 진행)
  python scripts/train.py --tag first
  # 학교 PC 8시간 세션: 이전 체크포인트 이어받기
  python scripts/train.py --resume runs/<이전 실행>/checkpoints/latest.zip --override configs/school_pc.yaml
  # 특정 단계만 고정 학습 (실험 비교용)
  python scripts/train.py --stage 5 --fixed-stage --set ppo.ent_coef=0.005
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from spider_rl.callbacks import CheckpointAndTimeLimitCallback, CurriculumLoggerCallback  # noqa: E402
from spider_rl.config import ROOT, load_config, save_config  # noqa: E402
from spider_rl.env import SpiderEnv  # noqa: E402


def make_env(cfg, rank, seed, stage, fixed_stage):
    def _init():
        env = SpiderEnv(cfg, stage=stage, fixed_stage=fixed_stage)
        env.reset(seed=seed + rank)
        return env
    return _init


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    # 내장(UHD 770)+외장(RTX 3050) 공존 PC: CUDA 장치는 NVIDIA만 잡힌다. 소형 MLP는 CPU가 더 빠른 경우가 많음.
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="ESP32 거미형 로봇 PPO 학습")
    ap.add_argument("--config", default=None, help="기본 설정 파일 (기본: configs/default.yaml)")
    ap.add_argument("--override", action="append", default=[], help="덮어쓸 설정 파일 (여러 번 가능)")
    ap.add_argument("--set", action="append", default=[], help="개별 설정 a.b.c=값 (여러 번 가능)")
    ap.add_argument("--tag", default="", help="실행 이름 꼬리표")
    ap.add_argument("--run-name", default=None, help="실행 폴더 이름을 직접 지정 (관제 대시보드가 사용)")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--resume", default=None, help="이어받을 체크포인트 .zip")
    ap.add_argument("--stage", type=int, default=None, help="시작 커리큘럼 단계")
    ap.add_argument("--fixed-stage", action="store_true", help="승급 없이 해당 단계만 학습")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--timesteps", type=float, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dummy", action="store_true", help="SubprocVecEnv 대신 단일 프로세스 (디버깅)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override, args.set)
    pc, tc = cfg["ppo"], cfg["train"]
    n_envs = args.n_envs or pc["n_envs"]
    total = int(args.timesteps or tc["total_timesteps"])
    max_hours = args.max_hours if args.max_hours is not None else tc["max_hours"]
    device = pick_device(args.device or pc["device"])
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 4)))

    # ---- 실행 폴더 ----
    host = socket.gethostname().replace(" ", "_")
    name = args.run_name or (time.strftime("%Y%m%d-%H%M%S") + f"_{host}" + (f"_{args.tag}" if args.tag else ""))
    run_dir = os.path.join(args.runs_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    # ---- 커리큘럼 시작점 ----
    start_stage, start_prog, stage_eps = 0, 0.0, 0
    if args.resume:
        cj = args.resume.replace(".zip", "_curriculum.json")
        if os.path.exists(cj):
            with open(cj, encoding="utf-8") as f:
                st = json.load(f)
            start_stage, start_prog, stage_eps = st["stage"], st["progress"], st["stage_episodes"]
    if args.stage is not None:
        start_stage, start_prog, stage_eps = args.stage, 0.0, 0

    meta = {"host": host, "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "device": device, "n_envs": n_envs, "resume": args.resume,
            "start_stage": start_stage, "cmd": " ".join(sys.argv)}
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

    # ---- 환경 ----
    seed = tc["seed"]
    set_random_seed(seed)
    fns = [make_env(cfg, i, seed, start_stage, args.fixed_stage) for i in range(n_envs)]
    venv = DummyVecEnv(fns) if (args.dummy or n_envs == 1) else SubprocVecEnv(fns, start_method="spawn")
    venv = VecMonitor(venv)
    vecnorm_path = args.resume.replace(".zip", "_vecnormalize.pkl") if args.resume else None
    if pc["normalize_obs"] or pc["normalize_reward"]:
        if vecnorm_path and os.path.exists(vecnorm_path):
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = True
        else:
            venv = VecNormalize(venv, norm_obs=pc["normalize_obs"], norm_reward=pc["normalize_reward"],
                                clip_obs=10.0, gamma=pc["gamma"])

    # ---- 모델 ----
    lr = pc["learning_rate"]
    lr_sched = (lambda p: lr * p) if pc["lr_schedule"] == "linear" else lr
    act = {"elu": torch.nn.ELU, "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[pc["activation"]]
    policy_kwargs = dict(net_arch=dict(pi=pc["net_arch"]["pi"], vf=pc["net_arch"]["vf"]),
                         activation_fn=act, log_std_init=pc["log_std_init"])
    if args.resume:
        model = PPO.load(args.resume, env=venv, device=device,
                         custom_objects={"learning_rate": lr_sched, "ent_coef": pc["ent_coef"]})
    else:
        model = PPO("MlpPolicy", venv, learning_rate=lr_sched, n_steps=pc["n_steps"], batch_size=pc["batch_size"],
                    n_epochs=pc["n_epochs"], gamma=pc["gamma"], gae_lambda=pc["gae_lambda"],
                    clip_range=pc["clip_range"], ent_coef=pc["ent_coef"], vf_coef=pc["vf_coef"],
                    max_grad_norm=pc["max_grad_norm"], target_kl=pc["target_kl"],
                    policy_kwargs=policy_kwargs, device=device, seed=seed, verbose=0)
    model.set_logger(configure(run_dir, ["stdout", "csv", "tensorboard"]))

    cur_cb = CurriculumLoggerCallback(run_dir, cfg, start_stage, start_prog, stage_eps, args.fixed_stage)
    ck_cb = CheckpointAndTimeLimitCallback(run_dir, tc["checkpoint_every"], max_hours, cur_cb, total)
    print(f"[학습 시작] run_dir={run_dir}  envs={n_envs}  device={device}  stage={start_stage}")
    try:
        model.learn(total_timesteps=total, callback=CallbackList([cur_cb, ck_cb]),
                    log_interval=tc["log_every_rollouts"], reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print("[중단] Ctrl+C → 저장합니다")
        ck_cb.save("latest")
        ck_cb.state = "interrupted"
        ck_cb._write_status()
    finally:
        venv.close()
    print(f"[완료] 결과: {run_dir}\n  그래프: python scripts/plot_report.py {run_dir}")


if __name__ == "__main__":
    main()
