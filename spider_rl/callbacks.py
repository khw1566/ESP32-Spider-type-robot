"""학습 콜백: 커리큘럼 승급, 보고서용 상세 로깅, 체크포인트, 시간 제한."""
from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .env import REWARD_TERMS

EPISODE_FIELDS = (["timesteps", "wall_time", "env_id", "cur_stage"] +
                  ["stage", "stage_name", "progress", "rand_level", "n_obstacles", "step_height", "prone_start",
                   "success", "reached", "fell", "timeout", "ep_len", "ep_time", "init_dist", "final_dist",
                   "min_dist", "goal_progress", "displacement", "mean_speed", "energy", "stand_time",
                   "act_latency", "sens_latency", "slip_deg_mean", "return"] +
                  [f"r_{k}" for k in REWARD_TERMS])


def _try_import_psutil():
    try:
        import psutil  # noqa
        return psutil
    except ImportError:
        return None


class CurriculumLoggerCallback(BaseCallback):
    """
    - 에피소드 종료 정보를 모아 episodes.csv 에 한 줄씩 기록 (히트맵·분포 그래프용)
    - 롤아웃마다 성공률/넘어짐/보상 항목별 평균 등을 SB3 로거(TensorBoard+CSV)에 기록
    - 현재 단계 성공률이 기준을 넘으면 다음 단계로 승급, 단계 내 진행도(난이도 램프) 갱신
    """

    def __init__(self, run_dir: str, cfg: dict, start_stage: int = 0, start_progress: float = 0.0,
                 stage_episodes: int = 0, fixed_stage: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.cfg = cfg
        cur = cfg["curriculum"]
        self.stages = cur["stages"]
        self.window = cur["window"]
        self.ramp = cur["ramp_episodes"]
        self.stage = start_stage
        self.stage_episodes = stage_episodes
        self.progress = start_progress
        self.fixed_stage = fixed_stage
        self.recent = deque(maxlen=self.window)          # 현재 단계(재학습 제외) 에피소드
        self.recent_all = defaultdict(lambda: deque(maxlen=self.window))  # 단계별
        self.buffer = []                                 # 이번 롤아웃 에피소드
        self.t0 = time.time()
        self.psutil = _try_import_psutil()
        ep_path = os.path.join(run_dir, "episodes.csv")
        new = not os.path.exists(ep_path)
        self.ep_file = open(ep_path, "a", newline="", encoding="utf-8")
        self.ep_writer = csv.DictWriter(self.ep_file, fieldnames=EPISODE_FIELDS, extrasaction="ignore")
        if new:
            self.ep_writer.writeheader()
        ev_path = os.path.join(run_dir, "curriculum_events.csv")
        new = not os.path.exists(ev_path)
        self.ev_file = open(ev_path, "a", newline="", encoding="utf-8")
        self.ev_writer = csv.writer(self.ev_file)
        if new:
            self.ev_writer.writerow(["timesteps", "wall_time", "event", "stage", "stage_name", "success_rate"])

    # ------------------------------------------------------------------
    def _push_curriculum(self):
        self.training_env.env_method("set_curriculum", self.stage, self.progress)

    def _on_training_start(self) -> None:
        self._push_curriculum()
        self.ev_writer.writerow([self.num_timesteps, 0, "start", self.stage, self.stages[self.stage]["name"], ""])
        self.ev_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            st = info.get("ep_stats")
            if st is None:
                continue
            row = dict(st)
            row.update(timesteps=self.num_timesteps, wall_time=round(time.time() - self.t0, 1), env_id=i,
                       cur_stage=self.stage)
            self.ep_writer.writerow(row)
            self.buffer.append(st)
            self.recent_all[st["stage"]].append(st)
            if st["stage"] == self.stage:
                self.recent.append(st)
                self.stage_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        log = self.logger
        self.ep_file.flush()
        eps = self.buffer
        self.buffer = []
        elapsed = time.time() - self.t0

        # ---- 커리큘럼 ----
        sr = float(np.mean([e["success"] for e in self.recent])) if self.recent else 0.0
        self.progress = min(1.0, self.stage_episodes / max(1, self.ramp))
        promote_at = self.stages[self.stage].get("promote_success", 0.8)
        if (not self.fixed_stage and self.stage < len(self.stages) - 1 and len(self.recent) >= self.window
                and sr >= promote_at and self.progress >= 0.3):
            self.ev_writer.writerow([self.num_timesteps, round(elapsed, 1), "promote", self.stage,
                                     self.stages[self.stage]["name"], round(sr, 4)])
            self.ev_file.flush()
            if self.verbose:
                print(f"[커리큘럼] {self.stages[self.stage]['name']} 성공률 {sr:.2f} → "
                      f"{self.stages[self.stage + 1]['name']} 단계로 승급 (t={self.num_timesteps})")
            self.stage += 1
            self.stage_episodes = 0
            self.progress = 0.0
            self.recent.clear()
        self._push_curriculum()

        log.record("curriculum/stage", self.stage)
        log.record("curriculum/stage_progress", self.progress)
        log.record("curriculum/stage_episodes", self.stage_episodes)
        log.record("curriculum/success_rate_window", sr)

        # ---- 성능 지표 ----
        if eps:
            cur = [e for e in eps if e["stage"] == self.stage] or eps
            def mean(key, src=eps):
                return float(np.mean([e[key] for e in src]))
            log.record("perf/success_rate", mean("success", cur))
            log.record("perf/reach_rate", mean("reached", cur))
            log.record("perf/fall_rate", mean("fell", cur))
            log.record("perf/timeout_rate", mean("timeout", cur))
            log.record("perf/ep_return_mean", mean("return", cur))
            log.record("perf/ep_return_max", float(np.max([e["return"] for e in cur])))
            log.record("perf/ep_return_min", float(np.min([e["return"] for e in cur])))
            log.record("perf/ep_len_mean", mean("ep_len", cur))
            log.record("perf/final_dist_mean", mean("final_dist", cur))
            log.record("perf/goal_progress_mean", mean("goal_progress", cur))
            log.record("perf/mean_speed", mean("mean_speed", cur))
            log.record("perf/energy_per_ep", mean("energy", cur))
            st = [e["stand_time"] for e in cur if e["stand_time"] >= 0]
            log.record("perf/stand_up_rate", len(st) / len(cur))
            if st:
                log.record("perf/stand_up_time", float(np.mean(st)))
            log.record("perf/slip_deg_mean", mean("slip_deg_mean", cur))
            log.record("perf/episodes_in_rollout", len(eps))
            for k in REWARD_TERMS:
                log.record(f"reward_terms/{k}", mean(f"r_{k}", cur))
                log.record(f"reward_terms_per_step/{k}", float(np.mean([e[f"r_{k}"] / max(1, e["ep_len"]) for e in cur])))
        for s, dq in self.recent_all.items():
            if dq:
                log.record(f"stage_success/{s}_{self.stages[s]['name']}", float(np.mean([e["success"] for e in dq])))

        # ---- 시스템 ----
        log.record("system/elapsed_hours", elapsed / 3600)
        log.record("system/steps_per_sec_total", self.num_timesteps / max(elapsed, 1e-6))
        if self.psutil is not None:
            log.record("system/cpu_percent", self.psutil.cpu_percent())
            log.record("system/ram_percent", self.psutil.virtual_memory().percent)
        try:
            import torch
            if torch.cuda.is_available():
                log.record("system/gpu_mem_mb", torch.cuda.memory_allocated() / 2 ** 20)
        except Exception:
            pass

    def state(self) -> dict:
        return {"stage": self.stage, "progress": self.progress, "stage_episodes": self.stage_episodes,
                "timesteps": int(self.num_timesteps)}

    def _on_training_end(self) -> None:
        self.ev_writer.writerow([self.num_timesteps, round(time.time() - self.t0, 1), "end", self.stage,
                                 self.stages[self.stage]["name"], ""])
        self.ep_file.close()
        self.ev_file.close()


class CheckpointAndTimeLimitCallback(BaseCallback):
    """주기적 체크포인트(모델+VecNormalize+커리큘럼 상태) 저장, 시간 제한 도달 시 저장 후 종료."""

    def __init__(self, run_dir: str, every: int, max_hours: float, curriculum: CurriculumLoggerCallback,
                 verbose: int = 1):
        super().__init__(verbose)
        self.dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.dir, exist_ok=True)
        self.every = every
        self.max_sec = max_hours * 3600 if max_hours and max_hours > 0 else None
        self.cur = curriculum
        self.t0 = time.time()
        self.last = 0
        self.hist_path = os.path.join(run_dir, "checkpoints.csv")

    def save(self, tag: str):
        base = os.path.join(self.dir, tag)
        self.model.save(base + ".zip")
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(base + "_vecnormalize.pkl")
        st = self.cur.state()
        with open(base + "_curriculum.json", "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
        new = not os.path.exists(self.hist_path)
        with open(self.hist_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timesteps", "wall_time", "tag", "stage", "progress"])
            w.writerow([self.num_timesteps, round(time.time() - self.t0, 1), tag, st["stage"], round(st["progress"], 3)])
        if self.verbose:
            print(f"[체크포인트] {base}.zip 저장 (stage={st['stage']})")

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.save(f"step_{self.num_timesteps:011d}")
            self.save("latest")
        if self.max_sec is not None and time.time() - self.t0 > self.max_sec:
            print("[시간 제한] 최대 학습 시간 도달 → 저장 후 종료")
            self.save("latest")
            return False
        return True

    def _on_training_end(self) -> None:
        self.save("final")
        self.save("latest")
