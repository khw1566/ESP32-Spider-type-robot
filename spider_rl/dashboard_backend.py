"""관제 대시보드의 GUI와 무관한 기능 (실행 목록, 상태 읽기, 학습 중 설정 변경, 학습 명령 생성).

학습 프로세스와는 실행 폴더의 파일로만 통신한다.
  control.json : 대시보드 → 학습 (시간 제한, 엔트로피 계수, 학습률, 지금 저장, 정지)
  status.json  : 학습 → 대시보드 (진행 상태)
따라서 대시보드를 껐다 켜도 학습은 계속되고, 다시 켜면 이어서 관제할 수 있다.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

from .config import ROOT

RUNS_DIR = os.path.join(ROOT, "runs")
STALE_SEC = 90          # status.json 이 이 시간 이상 갱신되지 않으면 '응답 없음'


def list_runs(runs_dir: str = RUNS_DIR) -> list[str]:
    """최근 수정 순으로 실행 폴더 이름 목록."""
    if not os.path.isdir(runs_dir):
        return []
    items = []
    for name in os.listdir(runs_dir):
        d = os.path.join(runs_dir, name)
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "config.yaml"))
                                 or os.path.exists(os.path.join(d, "train.log"))):
            items.append((os.path.getmtime(d), name))
    return [n for _, n in sorted(items, reverse=True)]


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_status(run_dir: str) -> dict | None:
    st = _read_json(os.path.join(run_dir, "status.json"))
    if st is None:
        return None
    st["stale"] = st.get("state") == "running" and time.time() - st.get("updated", 0) > STALE_SEC
    return st


def read_control(run_dir: str) -> dict:
    return _read_json(os.path.join(run_dir, "control.json")) or {}


def write_control(run_dir: str, **changes) -> dict:
    """control.json 에 바꿀 값만 병합해 원자적으로 기록."""
    path = os.path.join(run_dir, "control.json")
    ctrl = read_control(run_dir)
    ctrl.update(changes)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ctrl, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return ctrl


def tail(path: str, n: int = 40) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln for ln in data.splitlines() if not ln.startswith("|") and not ln.startswith("---")]
    return "\n".join(lines[-n:])


def latest_checkpoint(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "checkpoints", "latest.zip")
    return p if os.path.exists(p) else None


def fmt_duration(sec) -> str:
    if sec is None:
        return "제한 없음"
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m:02d}분" if h else f"{m}분 {s:02d}초"


def new_run_name(tag: str = "") -> str:
    host = socket.gethostname().replace(" ", "_")
    return time.strftime("%Y%m%d-%H%M%S") + f"_{host}" + (f"_{tag}" if tag else "")


def build_train_command(p: dict, python: str | None = None, runs_dir: str = RUNS_DIR) -> list[str]:
    """대시보드 입력값 → train.py 명령 (리스트)."""
    cmd = [python or sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", p["run_name"],
           "--runs-dir", runs_dir]
    if p.get("preset"):
        cmd += ["--override", os.path.join(ROOT, "configs", p["preset"] + ".yaml")]
    if p.get("n_envs"):
        cmd += ["--n-envs", str(int(p["n_envs"]))]
    if p.get("max_hours") is not None:
        cmd += ["--max-hours", str(float(p["max_hours"]))]
    if p.get("timesteps"):
        cmd += ["--timesteps", str(int(float(p["timesteps"])))]
    if p.get("resume"):
        cmd += ["--resume", p["resume"]]
    if p.get("stage") is not None and p.get("stage") != "":
        cmd += ["--stage", str(int(p["stage"]))]
    if p.get("fixed_stage"):
        cmd += ["--fixed-stage"]
    if p.get("device"):
        cmd += ["--device", p["device"]]
    for line in p.get("sets", []):
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            cmd += ["--set", line]
    return cmd


def launch_training(p: dict, runs_dir: str = RUNS_DIR) -> tuple[subprocess.Popen, str]:
    """학습을 대시보드와 독립된 프로세스로 실행 (대시보드를 닫아도 계속 돈다). 실행 폴더 경로 반환."""
    run_dir = os.path.join(runs_dir, p["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    cmd = build_train_command(p, runs_dir=runs_dir)
    log = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    kw = dict(cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    if os.name == "nt":
        kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    with open(os.path.join(run_dir, "launch_cmd.txt"), "w", encoding="utf-8") as f:
        f.write(subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(cmd, **kw)
    return proc, run_dir
