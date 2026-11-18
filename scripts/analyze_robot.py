"""로봇 설계 검토: 질량 예산, 지상고, 자세별 정적 관절 토크, 일어서기 속도별 탈조 여부.

보고서 '토크 검토' 절에 바로 쓸 수 있는 표(CSV)와 그래프를 만든다.
  python scripts/analyze_robot.py --out docs/robot_analysis
  python scripts/analyze_robot.py --set robot.link2_length=0.10   # 형상 변경 효과 비교
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from spider_rl.config import load_config  # noqa: E402
from spider_rl.env import SpiderEnv  # noqa: E402
from spider_rl.model_builder import output_torque, robot_mass_budget  # noqa: E402
from spider_rl.plotting import PALETTE, TEXT2, L, savefig, setup_style  # noqa: E402

KGFCM = 0.0980665  # 1 kgf·cm = 0.0980665 N·m


def static_knee_torques(cfg, angles):
    c = copy.deepcopy(cfg)
    c["robot"].update(step_loss=False, motor_holding_torque=10.0, max_joint_speed=1.0)
    env = SpiderEnv(c, stage=0, record=True, fixed_stage=True)
    env.cfg["curriculum"]["stages"][0]["randomization"] = [0.0, 0.0]
    env.reset(seed=0, options={"prone": True})
    a_nom, a_sc = env.act_nominal[1], env.act_scale[1]
    res = []
    for ang in angles:
        a = np.zeros(8)
        a[1::2] = (np.radians(ang) - a_nom) / a_sc
        for _ in range(60):
            env.step(np.clip(a, -1, 1))
        last = env.trace[-15:]
        tq = float(np.median([np.max(np.abs(t["torque"][1::2])) for t in last]))
        res.append((ang, float(np.degrees(last[-1]["q"][1])), tq, float(last[-1]["height"])))
    return res


def stand_trial(cfg, ramp_steps, lvl=0.0, seed=1):
    env = SpiderEnv(cfg, stage=0, record=True, fixed_stage=True)
    env.cfg["curriculum"]["stages"][0]["randomization"] = [lvl, lvl]
    env.reset(seed=seed, options={"prone": True})
    a_nom, a_sc = env.act_nominal[1], env.act_scale[1]
    lo = (0 - a_nom) / a_sc
    hi = (np.radians(90) - a_nom) / a_sc
    for i in range(int(2.5 * 30)):
        a = np.zeros(8)
        a[1::2] = lo + (hi - lo) * min(1.0, i / max(1, ramp_steps))
        env.step(a)
    return {"ramp_s": ramp_steps / 30, "height_cm": env.trace[-1]["height"] * 100,
            "slip_deg": float(np.degrees(np.abs(env.slip[1::2]).max())), "stood": env.stand_time >= 0,
            "max_torque": float(max(np.abs(t["torque"]).max() for t in env.trace))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="docs/robot_analysis")
    args = ap.parse_args()
    setup_style()
    cfg = load_config(args.config, None, args.set)
    r = cfg["robot"]
    os.makedirs(args.out, exist_ok=True)
    saved = []

    mb = robot_mass_budget(cfg)
    t_out = output_torque(r)
    L2, fr, sz = r["link2_length"], r["foot_radius"], r["body_size"][2]
    stand_h = L2 + fr - r["hip_z"]
    clearance = stand_h - sz / 2
    report = {"mass_budget_kg": mb, "output_torque_Nm": t_out, "output_torque_kgfcm": t_out / KGFCM,
              "stand_body_height_cm": stand_h * 100, "ground_clearance_cm": clearance * 100,
              "max_step_note": "지상고보다 높은 단차는 몸체가 걸림 → 링크2 연장 또는 hip_z 음수(고관절 하향) 필요"
              if clearance < 0.10 else "10 cm 단차까지 지상고 충분"}

    angles = list(range(0, 121, 10))
    st = static_knee_torques(cfg, angles)
    # 해석해: 몸 전체 무게를 n발이 나눠 지지, 무릎 토크 = (W/n) * L2 * cos(θ)
    W = mb["total"] * 9.81
    th = np.radians(np.linspace(0, 120, 121))
    req4 = W / 4 * L2 * np.abs(np.cos(th))
    req3 = W / 3 * L2 * np.abs(np.cos(th))
    report["analytic_knee_torque_4feet_max_Nm"] = float(req4.max())
    report["analytic_knee_torque_3feet_max_Nm"] = float(req3.max())
    report["torque_margin_3feet"] = t_out / float(req3.max())
    report["sim_knee_torque_peak_Nm"] = max(x[2] for x in st)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(np.degrees(th), req4 / KGFCM, color=PALETTE[0], label=L("필요 토크 해석해 (4발 지지)", "required, 4-foot support"))
    ax.plot(np.degrees(th), req3 / KGFCM, color=PALETTE[2], label=L("필요 토크 해석해 (3발 지지)", "required, 3-foot support"))
    ax.plot([x[1] for x in st], [x[2] / KGFCM for x in st], "o", color=PALETTE[3], ms=5,
            label=L("시뮬레이션 정지 자세 (발 마찰 내력 포함)", "simulated static pose (incl. friction)"))
    ax.axhline(t_out / KGFCM, color=PALETTE[1], lw=1.6, label=L("감속기 출력 정지 토크", "gearbox output holding torque"))
    ax.set_xlabel(L("무릎 각도 (deg, 0=엎드림)", "knee angle (deg, 0 = prone)"))
    ax.set_ylabel("kgf·cm")
    ax.set_title(L("무릎 각도별 필요 토크 vs 모터 출력", "Required knee torque vs available"))
    ax.legend(fontsize=7)
    savefig(fig, args.out, "knee_torque_vs_angle", saved)

    trials = [stand_trial(cfg, rs, lvl) for lvl in (0.0, 1.0) for rs in (6, 10, 15, 20, 30, 45, 60)]
    for t, lvl in zip(trials, [0.0] * 7 + [1.0] * 7):
        t["rand_level"] = lvl
    report["stand_trials"] = trials
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, lvl in enumerate((0.0, 1.0)):
        sub = [t for t in trials if t["rand_level"] == lvl]
        ax.plot([t["ramp_s"] for t in sub], [t["slip_deg"] for t in sub], "o-", color=PALETTE[i], ms=4,
                label=L(f"랜덤화 수준 {lvl:.0f}", f"randomization {lvl:.0f}"))
    ax.set_xlabel(L("엎드림→서기 명령 시간 (s)", "prone→stand command time (s)"))
    ax.set_ylabel(L("최대 탈조 오차 (deg)", "max step-loss error (deg)"))
    ax.set_title(L("일어서기 속도와 탈조", "Stand-up speed vs step loss"))
    ax.legend(fontsize=8)
    savefig(fig, args.out, "standup_speed_vs_step_loss", saved)

    with open(os.path.join(args.out, "robot_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "knee_torque_table.csv"), "w", encoding="utf-8") as f:
        f.write("cmd_deg,actual_deg,torque_Nm,torque_kgfcm,body_height_cm\n")
        for a, q, tq, h in st:
            f.write(f"{a},{q:.1f},{tq:.3f},{tq / KGFCM:.2f},{h * 100:.1f}\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("저장:", *saved, sep="\n  ")


if __name__ == "__main__":
    main()
