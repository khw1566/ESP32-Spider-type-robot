"""학습 결과 폴더(runs/...)에서 보고서용 그래프 PNG 를 일괄 생성한다.

사용법
  python scripts/plot_report.py runs/<실행폴더>                 # 단일 실행
  python scripts/plot_report.py runs/A runs/B --labels 기본 엔트로피↑  # 여러 실행 비교 그래프 추가
  python scripts/plot_report.py runs/A --out report_figs --smooth 10

출력: <out>/*.png 와 summary.csv (그래프에 쓰인 수치)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from spider_rl.env import REWARD_TERMS  # noqa: E402
from spider_rl.plotting import PALETTE, SEQ_CMAP, TEXT2, L, savefig, setup_style, smooth  # noqa: E402

TS = "time/total_timesteps"

TERM_KO = {"progress": "목표 접근", "reach": "도달 보너스", "heading": "방향 정렬", "upright": "수직 자세",
           "height": "서기 높이", "stand_bonus": "서기 유지", "alive": "생존", "body_contact": "몸체 접촉",
           "obstacle_contact": "장애물 충돌", "energy": "에너지", "torque": "토크", "action_rate": "행동 변화",
           "joint_limit": "관절 한계", "feet_air_time": "발 들기", "fall": "뒤집힘"}


def load_run(d):
    prog = pd.read_csv(os.path.join(d, "progress.csv")) if os.path.exists(os.path.join(d, "progress.csv")) else None
    if prog is not None and TS in prog:
        # train/* 지표는 다음 줄에 기록되므로 시간축 기준으로 앞으로 채움 없이 각 열을 따로 사용
        prog = prog.sort_values(TS)
    eps = pd.read_csv(os.path.join(d, "episodes.csv")) if os.path.exists(os.path.join(d, "episodes.csv")) else None
    ev = pd.read_csv(os.path.join(d, "curriculum_events.csv")) if os.path.exists(
        os.path.join(d, "curriculum_events.csv")) else None
    return prog, eps, ev


def series(prog, col, k=1):
    if prog is None or col not in prog:
        return None, None
    df = prog[[TS, col]].dropna()
    if df.empty:
        return None, None
    return df[TS].values, smooth(df[col].values, k)


def mark_stages(ax, ev):
    if ev is None:
        return
    for _, r in ev[ev["event"] == "promote"].iterrows():
        ax.axvline(r["timesteps"], color=TEXT2, lw=0.8, ls="--", alpha=0.6)
        ax.text(r["timesteps"], ax.get_ylim()[1], f" →{int(r['stage']) + 1}", fontsize=7, color=TEXT2,
                va="top", ha="left")


def line_plot(prog, ev, cols, labels, title, ylabel, out, name, saved, k, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    n = 0
    for i, (c, lab) in enumerate(zip(cols, labels)):
        x, y = series(prog, c, k)
        if x is None:
            continue
        ax.plot(x, y, color=PALETTE[i % 8], label=lab)
        n += 1
    if n == 0:
        plt.close(fig)
        return
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel(L("학습 스텝", "timesteps"))
    ax.set_ylabel(ylabel)
    mark_stages(ax, ev)
    if n > 1:
        ax.legend(fontsize=8)
    savefig(fig, out, name, saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()
    setup_style()
    k = args.smooth
    main_run = args.runs[0]
    out = args.out or os.path.join(main_run, "figures")
    saved: list[str] = []
    prog, eps, ev = load_run(main_run)
    stages_names = None
    if eps is not None and len(eps):
        stages_names = eps.groupby("stage")["stage_name"].first().to_dict()

    # ---------------- 성능 ----------------
    line_plot(prog, ev, ["perf/success_rate", "curriculum/success_rate_window"],
              [L("롤아웃 성공률", "rollout success"), L("최근 윈도 성공률(승급 기준)", "window success (promotion)")],
              L("성공률 (현재 커리큘럼 단계)", "Success rate (current stage)"), L("비율", "rate"),
              out, "01_success_rate", saved, k, (-0.02, 1.02))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x, y = series(prog, "perf/ep_return_mean", k)
    if x is not None:
        _, lo = series(prog, "perf/ep_return_min", k)
        _, hi = series(prog, "perf/ep_return_max", k)
        ax.fill_between(x, lo, hi, color=PALETTE[0], alpha=0.15, lw=0, label=L("최소~최대", "min-max"))
        ax.plot(x, y, color=PALETTE[0], label=L("평균", "mean"))
        ax.set_title(L("에피소드 보상 (평균/최대/최소)", "Episode return (mean/min/max)"))
        ax.set_xlabel(L("학습 스텝", "timesteps"))
        ax.set_ylabel(L("보상", "return"))
        mark_stages(ax, ev)
        ax.legend(fontsize=8)
        savefig(fig, out, "02_episode_return", saved)
    else:
        plt.close(fig)
    line_plot(prog, ev, ["perf/ep_len_mean"], [""], L("에피소드 길이", "Episode length"), L("스텝 (30 Hz)", "steps"),
              out, "03_episode_length", saved, k)
    line_plot(prog, ev, ["perf/reach_rate", "perf/fall_rate", "perf/timeout_rate"],
              [L("목표 도달", "reached"), L("뒤집힘", "fell"), L("시간 초과", "timeout")],
              L("에피소드 종료 원인 비율", "Episode outcome rates"), L("비율", "rate"), out, "04_outcome_rates",
              saved, k, (-0.02, 1.02))
    line_plot(prog, ev, ["perf/final_dist_mean"], [""], L("종료 시 목표까지 남은 거리", "Final distance to goal"),
              "m", out, "05_final_distance", saved, k)
    line_plot(prog, ev, ["perf/goal_progress_mean"], [""], L("에피소드당 목표 방향 이동 거리", "Goal progress per ep"),
              "m", out, "06_goal_progress", saved, k)
    line_plot(prog, ev, ["perf/mean_speed"], [""], L("평균 목표 방향 속도", "Mean speed toward goal"), "m/s",
              out, "07_speed", saved, k)
    line_plot(prog, ev, ["perf/stand_up_rate"], [""], L("일어서기 성공 비율", "Stand-up rate"), L("비율", "rate"),
              out, "08_stand_up_rate", saved, k, (-0.02, 1.02))
    line_plot(prog, ev, ["perf/stand_up_time"], [""], L("일어서기 소요 시간", "Stand-up time"), L("초", "s"),
              out, "09_stand_up_time", saved, k)
    line_plot(prog, ev, ["perf/energy_per_ep"], [""], L("에피소드당 기계적 에너지", "Mechanical energy per ep"),
              "J", out, "10_energy", saved, k)
    line_plot(prog, ev, ["perf/slip_deg_mean"], [""], L("탈조 누적 오차 (관절 평균)", "Step-loss drift (mean)"),
              "deg", out, "11_step_loss", saved, k)
    line_plot(prog, ev, ["rollout/ep_rew_mean"], [""], L("정규화 전 에피소드 보상 (Monitor)", "Monitor ep reward"),
              L("보상", "return"), out, "12_monitor_reward", saved, k)

    # ---------------- PPO 내부 지표 (패널 분리, 이중축 없음) ----------------
    ppo = [("train/entropy_loss", L("엔트로피 손실", "entropy loss")),
           ("train/std", L("행동 표준편차", "action std")),
           ("train/policy_gradient_loss", L("정책 손실", "policy loss")),
           ("train/value_loss", L("가치 손실", "value loss")),
           ("train/approx_kl", "approx KL"), ("train/clip_fraction", L("클립 비율", "clip fraction")),
           ("train/explained_variance", L("설명 분산", "explained variance")),
           ("train/learning_rate", L("학습률", "learning rate")), ("train/loss", L("전체 손실", "total loss"))]
    avail = [(c, t) for c, t in ppo if prog is not None and c in prog and prog[c].notna().any()]
    if avail:
        n = len(avail)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.8 * rows), squeeze=False)
        for ax, (c, t) in zip(axes.flat, avail):
            x, y = series(prog, c, k)
            ax.plot(x, y, color=PALETTE[0])
            ax.set_title(t, fontsize=10)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(L("PPO 학습 지표", "PPO training metrics"))
        savefig(fig, out, "13_ppo_metrics", saved)
        for c, t in avail:   # 개별 그래프도 저장 (보고서에 하나씩 넣기 편하게)
            line_plot(prog, ev, [c], [""], t, "", out, "13_ppo_" + c.split("/")[1], saved, k)

    # ---------------- 보상 분해 ----------------
    cols_ = [f"reward_terms_per_step/{t}" for t in REWARD_TERMS]
    have = [c for c in cols_ if prog is not None and c in prog and prog[c].notna().any()]
    if have:
        df = prog[[TS] + have].dropna()
        mags = df[have].abs().mean().sort_values(ascending=False)
        top = [c for c in mags.index if mags[c] > 1e-9][:7]
        rest = [c for c in have if c not in top]
        x = df[TS].values
        data = {c: smooth(df[c].values, k) for c in top}
        if rest:
            data["other"] = smooth(df[rest].sum(axis=1).values, k)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        pos_base = np.zeros(len(x))
        neg_base = np.zeros(len(x))
        for i, (c, y) in enumerate(data.items()):
            name = TERM_KO.get(c.split("/")[-1], L("기타", "other")) if setup_style() else c.split("/")[-1]
            yp, yn = np.clip(y, 0, None), np.clip(y, None, 0)
            ax.fill_between(x, pos_base, pos_base + yp, color=PALETTE[i % 8], lw=0, label=name)
            ax.fill_between(x, neg_base, neg_base + yn, color=PALETTE[i % 8], lw=0)
            pos_base += yp
            neg_base += yn
        ax.axhline(0, color=TEXT2, lw=0.8)
        ax.set_title(L("보상 항목별 기여도 (스텝당 평균, 누적 영역)", "Reward decomposition (per step, stacked)"))
        ax.set_xlabel(L("학습 스텝", "timesteps"))
        ax.set_ylabel(L("보상/스텝", "reward/step"))
        mark_stages(ax, ev)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        savefig(fig, out, "14_reward_decomposition", saved)

        n = len(have)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2.4 * rows), squeeze=False)
        for ax, c in zip(axes.flat, have):
            xx, yy = series(prog, c.replace("reward_terms_per_step", "reward_terms"), k)
            if xx is not None:
                ax.plot(xx, yy, color=PALETTE[0])
            t = c.split("/")[-1]
            ax.set_title(TERM_KO.get(t, t) if setup_style() else t, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(L("보상 항목별 에피소드 합계", "Reward terms per episode"))
        savefig(fig, out, "15_reward_terms_grid", saved)

    # ---------------- 커리큘럼 ----------------
    x, y = series(prog, "curriculum/stage", 1)
    if x is not None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.step(x, y, where="post", color=PALETTE[0])
        if stages_names:
            ax.set_yticks(list(stages_names.keys()))
            ax.set_yticklabels([f"{s}:{n}" for s, n in stages_names.items()], fontsize=7)
        ax.set_title(L("커리큘럼 단계 진행", "Curriculum stage"))
        ax.set_xlabel(L("학습 스텝", "timesteps"))
        savefig(fig, out, "16_curriculum_stage", saved)
    stage_cols = [c for c in (prog.columns if prog is not None else []) if c.startswith("stage_success/")]
    if stage_cols:
        stage_cols = sorted(stage_cols, key=lambda c: int(c.split("/")[1].split("_")[0]))
        line_plot(prog, ev, stage_cols, [c.split("/")[1] for c in stage_cols],
                  L("단계별 성공률 (재학습 에피소드 포함)", "Per-stage success rate"), L("비율", "rate"), out,
                  "17_stage_success", saved, k, (-0.02, 1.02))

    # ---------------- 에피소드 단위 분석 (episodes.csv) ----------------
    if eps is not None and len(eps):
        g = eps.groupby("stage").agg(success=("success", "mean"), fell=("fell", "mean"), timeout=("timeout", "mean"),
                                     reached=("reached", "mean"), n=("success", "size")).reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.6))
        names = [f"{int(s)}:{stages_names.get(s, s)}" for s in g["stage"]]
        xs = np.arange(len(g))
        other = 1 - g["fell"] - g["timeout"]
        bottom = np.zeros(len(g))
        for i, (vals, lab) in enumerate([(other, L("종료(도달 등)", "terminated (goal)")), (g["timeout"], L("시간 초과", "timeout")),
                                         (g["fell"], L("뒤집힘", "fell"))]):
            ax.bar(xs, vals, bottom=bottom, color=PALETTE[i], width=0.6, label=lab, edgecolor="white", linewidth=1.5)
            bottom += vals.values
        ax.plot(xs, g["success"], "o", color="#0b0b0b", ms=6, label=L("성공률", "success"))
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(L("단계별 에피소드 결과 분포 (전체 기간)", "Episode outcomes per stage"))
        ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        savefig(fig, out, "18_outcomes_by_stage", saved)
        g.to_csv(os.path.join(out, "summary_by_stage.csv"), index=False)

        ge = eps[eps["init_dist"] > 0]
        if len(ge) > 20:
            bins = np.arange(0.75, 4.51, 0.5)
            ge = ge.assign(dbin=pd.cut(ge["init_dist"], bins))
            piv = ge.pivot_table(index="stage", columns="dbin", values="success", aggfunc="mean", observed=False)
            fig, ax = plt.subplots(figsize=(8, 3.8))
            im = ax.imshow(piv.values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1)
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels([f"{b.left:.1f}-{b.right:.1f}" for b in piv.columns], fontsize=7)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([f"{s}:{stages_names.get(s, s)}" for s in piv.index], fontsize=7)
            for (i, j), v in np.ndenumerate(piv.values):
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 else "#0b0b0b")
            ax.set_xlabel(L("초기 목표 거리 (m)", "initial goal distance (m)"))
            ax.set_title(L("단계 × 목표 거리별 성공률 히트맵", "Success heatmap: stage × goal distance"))
            ax.grid(False)
            fig.colorbar(im, ax=ax, label=L("성공률", "success"))
            savefig(fig, out, "19_heatmap_stage_distance", saved)

        gt = eps[(eps["step_height"] > 0) | (eps["n_obstacles"] > 0)]
        if len(gt) > 20:
            gt = gt.assign(hbin=pd.cut(gt["step_height"], [-0.001, 0.001, 0.06, 0.07, 0.08, 0.09, 0.101]))
            piv = gt.pivot_table(index="n_obstacles", columns="hbin", values="success", aggfunc="mean", observed=False)
            fig, ax = plt.subplots(figsize=(7, 3.6))
            im = ax.imshow(piv.values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1, origin="lower")
            labs = [L("없음", "none")] + [f"{b.left * 100:.0f}-{b.right * 100:.0f}cm" for b in piv.columns[1:]]
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels(labs, fontsize=7)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([int(v) for v in piv.index], fontsize=7)
            for (i, j), v in np.ndenumerate(piv.values):
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 else "#0b0b0b")
            ax.set_xlabel(L("단차 높이", "step height"))
            ax.set_ylabel(L("장애물 개수", "obstacles"))
            ax.set_title(L("지형 난이도별 성공률", "Success by terrain difficulty"))
            ax.grid(False)
            fig.colorbar(im, ax=ax, label=L("성공률", "success"))
            savefig(fig, out, "20_heatmap_terrain", saved)

        for col, lab, name in [("act_latency", L("행동(통신) 지연 스텝", "action latency steps"), "21_success_vs_act_latency"),
                               ("sens_latency", L("센서 지연 스텝", "sensor latency steps"), "22_success_vs_sens_latency")]:
            sub = eps[eps["stage"] >= 3] if (eps["stage"] >= 3).any() else eps
            gg = sub.groupby(col)["success"].agg(["mean", "size"]).reset_index()
            if len(gg) > 1:
                fig, ax = plt.subplots(figsize=(5, 3.2))
                ax.bar(gg[col].astype(str), gg["mean"], color=PALETTE[0], width=0.55)
                for i, r in gg.iterrows():
                    ax.text(i, r["mean"] + 0.02, f"{r['mean']:.2f}\n(n={int(r['size'])})", ha="center", fontsize=7)
                ax.set_ylim(0, 1.15)
                ax.set_xlabel(lab)
                ax.set_ylabel(L("성공률", "success"))
                ax.set_title(L("지연에 따른 성공률", "Success vs latency"))
                savefig(fig, out, name, saved)

        if (eps["stand_time"] > 0).any():
            fig, ax = plt.subplots(figsize=(6, 3.2))
            st = eps.loc[eps["stand_time"] > 0, "stand_time"]
            ax.hist(st, bins=30, color=PALETTE[0], edgecolor="white")
            ax.set_xlabel(L("일어서기 시간 (s)", "stand-up time (s)"))
            ax.set_ylabel(L("에피소드 수", "episodes"))
            ax.set_title(L("엎드린 자세 → 서기 소요 시간 분포", "Stand-up time distribution"))
            savefig(fig, out, "23_stand_up_hist", saved)

        fig, ax = plt.subplots(figsize=(7, 3.4))
        w = max(20, len(eps) // 100)
        ax.plot(eps["timesteps"], eps["success"].rolling(w, min_periods=1).mean(), color=PALETTE[0],
                label=L("성공", "success"))
        ax.plot(eps["timesteps"], eps["fell"].rolling(w, min_periods=1).mean(), color=PALETTE[1],
                label=L("뒤집힘", "fell"))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(L("학습 스텝", "timesteps"))
        ax.set_title(L(f"에피소드 이동평균 (창 {w}, 재학습 포함)", f"Episode rolling mean (w={w})"))
        mark_stages(ax, ev)
        ax.legend(fontsize=8)
        savefig(fig, out, "24_episode_rolling", saved)

    # ---------------- 시스템 ----------------
    line_plot(prog, ev, ["time/fps"], [""], L("학습 속도", "Training speed"), L("스텝/초", "steps/s"), out,
              "25_fps", saved, 1)
    line_plot(prog, ev, ["system/cpu_percent", "system/ram_percent"], ["CPU %", "RAM %"],
              L("시스템 사용률", "System usage"), "%", out, "26_system", saved, 1)
    if os.path.exists(os.path.join(main_run, "checkpoints.csv")):
        ck = pd.read_csv(os.path.join(main_run, "checkpoints.csv"))
        ck = ck[~ck["tag"].isin(["latest"])]
        if len(ck):
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.plot(ck["wall_time"] / 3600, ck["timesteps"], "o-", color=PALETTE[0], ms=4)
            ax.set_xlabel(L("경과 시간 (h)", "elapsed (h)"))
            ax.set_ylabel(L("학습 스텝", "timesteps"))
            ax.set_title(L("체크포인트 이력", "Checkpoint history"))
            savefig(fig, out, "27_checkpoints", saved)

    # ---------------- 여러 실행 비교 ----------------
    if len(args.runs) > 1:
        labels = args.labels if args.labels and len(args.labels) == len(args.runs) else \
            [os.path.basename(os.path.normpath(r)) for r in args.runs]
        runs = [(lab, load_run(r)) for lab, r in zip(labels, args.runs)][:8]
        for col, title, name, ylim in [("perf/success_rate", L("성공률 비교", "Success rate"), "30_compare_success", (-0.02, 1.02)),
                                       ("perf/ep_return_mean", L("에피소드 보상 비교", "Return"), "31_compare_return", None),
                                       ("curriculum/stage", L("커리큘럼 단계 비교", "Stage"), "32_compare_stage", None),
                                       ("train/entropy_loss", L("엔트로피 손실 비교", "Entropy loss"), "33_compare_entropy", None),
                                       ("time/fps", L("학습 속도 비교", "FPS"), "34_compare_fps", None)]:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            n = 0
            for i, (lab, (p, _, _)) in enumerate(runs):
                x, y = series(p, col, k)
                if x is not None:
                    ax.plot(x, y, color=PALETTE[i], label=lab)
                    n += 1
            if n:
                if ylim:
                    ax.set_ylim(*ylim)
                ax.set_title(title)
                ax.set_xlabel(L("학습 스텝", "timesteps"))
                ax.legend(fontsize=8)
                savefig(fig, out, name, saved)
            else:
                plt.close(fig)

    if prog is not None:
        prog.to_csv(os.path.join(out, "progress_export.csv"), index=False)
    print(f"[그래프] {len(saved)}개 저장 → {out}")
    for s in saved:
        print("  ", os.path.basename(s))


if __name__ == "__main__":
    main()
