"""학습 중간 보고서 (HTML 한 파일, 이미지 내장) 생성.

학습 중에는 N판마다 자동으로 호출된다 (train.report_every_episodes, 대시보드에서 변경 가능).
직접 실행도 가능:
  python scripts/make_report.py runs/<실행>                       # 지금 시점 보고서
  python scripts/make_report.py runs/<실행> --replay runs/<실행>/replays/xxx.npz

출력: runs/<실행>/reports/report_0003_ep15000.html  +  reports/index.html (전체 보고서 목록·추이)
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from spider_rl.env import REWARD_TERMS  # noqa: E402
from spider_rl.model_builder import LEG_NAMES  # noqa: E402
from spider_rl.plotting import PALETTE, SEQ_CMAP, TEXT2, L, setup_style, smooth  # noqa: E402

TS = "time/total_timesteps"
TERM_KO = {"progress": "목표 접근", "reach": "도달 보너스", "heading": "방향 정렬", "upright": "수직 자세",
           "height": "서기 높이", "stand_bonus": "서기 유지", "alive": "생존", "body_contact": "몸체 접촉",
           "obstacle_contact": "장애물 충돌", "energy": "에너지", "torque": "토크", "action_rate": "행동 변화",
           "joint_limit": "관절 한계", "feet_air_time": "발 들기", "fall": "뒤집힘"}


# ----------------------------------------------------------------------
def fig_b64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_b64(arr) -> str:
    import imageio.v2 as imageio
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(x) -> str:
    return html.escape(str(x))


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def num(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def series(prog, col, k=1):
    if prog is None or col not in prog:
        return None, None
    df = prog[[TS, col]].dropna()
    if df.empty:
        return None, None
    return df[TS].values / 1e6, smooth(df[col].values, k)


def mark_stages(ax, ev):
    if ev is None:
        return
    for _, r in ev[ev["event"] == "promote"].iterrows():
        ax.axvline(r["timesteps"] / 1e6, color=TEXT2, lw=0.8, ls="--", alpha=0.6)


# ----------------------------------------------------------------------
# 3D 장면 이미지
# ----------------------------------------------------------------------
def render_scenes(rec, width=480, height=300):
    import mujoco
    from spider_rl.env import SpiderEnv
    from spider_rl.replay import apply_layout
    env = SpiderEnv(rec["config"])
    m, d = env.model, env.data
    apply_layout(m, rec["layout"])
    r = mujoco.Renderer(m, height=height, width=width)
    T = len(rec["qpos"])
    dt = rec["meta"]["dt"]
    frames = []
    for frac, name in [(0.0, "시작"), (0.33, "1/3 지점"), (0.66, "2/3 지점"), (1.0, "끝")]:
        i = min(T - 1, int(round(frac * (T - 1))))
        d.qpos[:] = rec["qpos"][i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="track")
        frames.append((f"{name} (t={i * dt:.1f}s)", img_b64(r.render())))
    top = mujoco.MjvCamera()
    top.type = mujoco.mjtCamera.mjCAMERA_FREE
    top.lookat[:] = [0, 0, 0]
    top.distance, top.elevation, top.azimuth = 6.2, -89.9, 90
    r2 = mujoco.Renderer(m, height=480, width=480)
    d.qpos[:] = rec["qpos"][-1]
    mujoco.mj_forward(m, d)
    r2.update_scene(d, camera=top)
    # 이동 궤적을 장면에 선으로 추가
    scn = r2.scene
    pts = rec["qpos"][::3, :3]
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.16, 0.47, 0.84, 1.0]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 4.0, a + [0, 0, 0.05], b + [0, 0, 0.05])
        scn.ngeom += 1
    top_img = img_b64(r2.render())
    r.close()
    r2.close()
    return frames, top_img


def episode_charts(rec):
    s = rec["series"]
    t = s["t"]
    out = {}
    con = s["contacts"].astype(bool)
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    for i, leg in enumerate(LEG_NAMES):
        on = con[:, i].astype(int)
        st = np.where(np.diff(np.r_[0, on]) == 1)[0]
        en = np.where(np.diff(np.r_[on, 0]) == -1)[0]
        ax.broken_barh([(t[a], t[b] - t[a] + rec["meta"]["dt"]) for a, b in zip(st, en)], (i - 0.35, 0.7),
                       color=PALETTE[0])
    ax.set_yticks(range(4))
    ax.set_yticklabels(LEG_NAMES)
    ax.set_xlabel(L("시간 (s)", "time (s)"))
    ax.set_title(L("보행 패턴 (막대 = 발 접지)", "Gait diagram"), fontsize=10)
    ax.grid(False)
    out["gait"] = fig_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharex=True)
    q = np.degrees(s["q"])
    for j, leg in enumerate(LEG_NAMES):
        axes[0].plot(t, q[:, 2 * j], color=PALETTE[j], lw=1.1, label=leg)
        axes[1].plot(t, q[:, 2 * j + 1], color=PALETTE[j], lw=1.1)
    axes[0].set_title(L("고관절 각도 (deg)", "hip (deg)"), fontsize=9)
    axes[1].set_title(L("무릎 각도 (deg)", "knee (deg)"), fontsize=9)
    axes[0].legend(fontsize=6, ncol=2)
    for ax in axes:
        ax.tick_params(labelsize=7)
    out["joints"] = fig_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axes[0].plot(t, s["height"] * 100, color=PALETTE[0])
    axes[0].set_title(L("몸체 높이 (cm)", "body height (cm)"), fontsize=9)
    axes[1].plot(t, s["goal_dist"], color=PALETTE[1])
    axes[1].set_title(L("목표까지 거리 (m)", "distance to goal (m)"), fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.set_xlabel(L("시간 (s)", "time (s)"), fontsize=8)
    out["height_goal"] = fig_b64(fig)

    if rec["rewards"]:
        fig, ax = plt.subplots(figsize=(6.4, 2.4))
        terms = sorted(rec["rewards"], key=lambda k: -abs(rec["rewards"][k].sum()))[:6]
        for i, k in enumerate(terms):
            ax.plot(t, np.cumsum(rec["rewards"][k]), color=PALETTE[i], lw=1.2, label=TERM_KO.get(k, k))
        ax.set_title(L("보상 항목 누적 (상위 6개)", "cumulative reward terms"), fontsize=9)
        ax.legend(fontsize=6, ncol=3)
        ax.tick_params(labelsize=7)
        out["rewards"] = fig_b64(fig)
    return out


# ----------------------------------------------------------------------
# 학습 곡선
# ----------------------------------------------------------------------
def training_charts(prog, ev, eps, k=5):
    out = {}
    if prog is None or TS not in prog:
        return out
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
    x, y = series(prog, "perf/success_rate", k)
    ax = axes[0, 0]
    if x is not None:
        ax.plot(x, y, color=PALETTE[0], label=L("롤아웃", "rollout"))
        x2, y2 = series(prog, "curriculum/success_rate_window", 1)
        if x2 is not None:
            ax.plot(x2, y2, color=PALETTE[1], label=L("승급 판단 창", "window"))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
    ax.set_title(L("성공률 (현재 단계)", "success rate"), fontsize=10)
    mark_stages(ax, ev)
    ax = axes[0, 1]
    x, y = series(prog, "perf/ep_return_mean", k)
    if x is not None:
        _, lo = series(prog, "perf/ep_return_min", k)
        _, hi = series(prog, "perf/ep_return_max", k)
        ax.fill_between(x, lo, hi, color=PALETTE[0], alpha=0.15, lw=0)
        ax.plot(x, y, color=PALETTE[0])
    ax.set_title(L("에피소드 보상 (평균, 최소~최대)", "episode return"), fontsize=10)
    mark_stages(ax, ev)
    ax = axes[1, 0]
    for i, (c, lab) in enumerate([("perf/reach_rate", L("도달", "reached")), ("perf/fall_rate", L("뒤집힘", "fell")),
                                  ("perf/timeout_rate", L("시간 초과", "timeout"))]):
        x, y = series(prog, c, k)
        if x is not None:
            ax.plot(x, y, color=PALETTE[i], label=lab)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.set_title(L("종료 원인 비율", "outcomes"), fontsize=10)
    ax.set_xlabel(L("백만 스텝", "M steps"))
    ax = axes[1, 1]
    x, y = series(prog, "curriculum/stage", 1)
    if x is not None:
        ax.step(x, y, where="post", color=PALETTE[0])
        ax.set_yticks(range(8))
        ax.set_ylim(-0.3, 7.3)
    ax.set_title(L("커리큘럼 단계", "curriculum stage"), fontsize=10)
    ax.set_xlabel(L("백만 스텝", "M steps"))
    for a in axes.flat:
        a.tick_params(labelsize=7)
    out["curves"] = fig_b64(fig)

    cols = [f"reward_terms_per_step/{t}" for t in REWARD_TERMS if f"reward_terms_per_step/{t}" in prog]
    if cols:
        df = prog[[TS] + cols].dropna()
        if len(df) > 1:
            mags = df[cols].abs().mean().sort_values(ascending=False)
            top = [c for c in mags.index if mags[c] > 1e-9][:7]
            rest = [c for c in cols if c not in top]
            fig, ax = plt.subplots(figsize=(10, 3.2))
            xx = df[TS].values / 1e6
            pb, nb = np.zeros(len(xx)), np.zeros(len(xx))
            items = [(TERM_KO.get(c.split("/")[-1], c), smooth(df[c].values, k)) for c in top]
            if rest:
                items.append((L("기타", "other"), smooth(df[rest].sum(axis=1).values, k)))
            for i, (name, yy) in enumerate(items):
                yp, yn = np.clip(yy, 0, None), np.clip(yy, None, 0)
                ax.fill_between(xx, pb, pb + yp, color=PALETTE[i % 8], lw=0, label=name)
                ax.fill_between(xx, nb, nb + yn, color=PALETTE[i % 8], lw=0)
                pb += yp
                nb += yn
            ax.axhline(0, color=TEXT2, lw=0.8)
            ax.legend(fontsize=7, ncol=8, loc="upper left")
            ax.set_title(L("보상 항목별 기여도 (스텝당, 위=보상 / 아래=패널티)", "reward decomposition"), fontsize=10)
            ax.set_xlabel(L("백만 스텝", "M steps"))
            ax.tick_params(labelsize=7)
            out["rewards"] = fig_b64(fig)

    ppo = [("train/entropy_loss", L("엔트로피 손실", "entropy")), ("train/std", L("행동 표준편차", "std")),
           ("train/approx_kl", "approx KL"), ("train/clip_fraction", L("클립 비율", "clip frac")),
           ("train/value_loss", L("가치 손실", "value loss")),
           ("train/explained_variance", L("설명 분산", "expl. var")), ("time/fps", L("학습 속도", "fps")),
           ("train/learning_rate", L("학습률", "lr"))]
    avail = [(c, t) for c, t in ppo if c in prog and prog[c].notna().sum() > 1]
    if avail:
        fig, axes = plt.subplots(2, 4, figsize=(10, 3.8))
        for ax, (c, t) in zip(axes.flat, avail):
            x, y = series(prog, c, k)
            ax.plot(x, y, color=PALETTE[0], lw=1.2)
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(avail):]:
            ax.axis("off")
        out["ppo"] = fig_b64(fig)

    if eps is not None and len(eps) > 30 and (eps["init_dist"] > 0).any():
        ge = eps[eps["init_dist"] > 0]
        bins = np.arange(0.75, 4.51, 0.5)
        ge = ge.assign(dbin=pd.cut(ge["init_dist"], bins))
        piv = ge.pivot_table(index="stage", columns="dbin", values="success", aggfunc="mean", observed=False)
        if piv.size:
            fig, ax = plt.subplots(figsize=(6.4, 2.8))
            ax.imshow(piv.values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1)
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels([f"{b.left:.1f}-{b.right:.1f}" for b in piv.columns], fontsize=7)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([str(s) for s in piv.index], fontsize=7)
            for (i, j), v in np.ndenumerate(piv.values):
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 else "#0b0b0b")
            ax.set_xlabel(L("목표 거리 (m)", "goal distance (m)"))
            ax.set_ylabel(L("단계", "stage"))
            ax.set_title(L("단계 × 목표 거리 성공률 (누적)", "success heatmap"), fontsize=10)
            ax.grid(False)
            out["heatmap"] = fig_b64(fig)
    return out


def trend_chart(hist):
    if len(hist) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
    x = [h["index"] for h in hist]
    axes[0].plot(x, [h["interval_success"] for h in hist], "o-", color=PALETTE[0], ms=4, label=L("성공률", "success"))
    axes[0].plot(x, [h["interval_fall"] for h in hist], "o-", color=PALETTE[1], ms=4, label=L("뒤집힘", "fell"))
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=7)
    axes[0].set_title(L("보고서별 구간 성공률·뒤집힘", "per-report success / fall"), fontsize=10)
    axes[1].step(x, [h["stage"] for h in hist], where="post", color=PALETTE[0])
    axes[1].set_title(L("보고서별 커리큘럼 단계", "stage per report"), fontsize=10)
    for ax in axes:
        ax.set_xlabel(L("보고서 번호", "report #"))
        ax.tick_params(labelsize=7)
    return fig_b64(fig)


# ----------------------------------------------------------------------
CSS = """
:root{--bg:#f6f6f4;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--good:#008300;--bad:#e34948}
@media (prefers-color-scheme: dark){:root{--bg:#141413;--card:#1f1f1d;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5;--good:#0ca30c;--bad:#e66767}
 img.chart{background:#fcfcfb;border-radius:6px}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif;line-height:1.5}
.wrap{max-width:1120px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--ink2);font-size:13px}.summary{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);padding:12px 16px;border-radius:8px;margin:16px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .l{font-size:12px;color:var(--ink2)}.kpi .v{font-size:24px;font-weight:700;margin-top:2px}.kpi .v.s{font-size:17px}.kpi .d{font-size:12px;margin-top:2px;color:var(--ink2)}
.up{color:var(--good)}.down{color:var(--bad)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
figure img{width:100%;display:block;border-radius:6px}figcaption{font-size:12px;color:var(--ink2);margin-top:4px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}th{background:var(--bg);font-weight:600;color:var(--ink2)}
td:first-child,th:first-child{text-align:left}.tablewrap{overflow-x:auto}
code{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:12px;word-break:break-all}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--line)}
a{color:var(--accent)}
"""


def kpi(label, value, delta=None, good_up=True, tip=""):
    d = ""
    if delta is not None and not (isinstance(delta, float) and math.isnan(delta)):
        cls = "up" if (delta > 0) == good_up and delta != 0 else ("down" if delta != 0 else "")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        d = f'<div class="d {cls}">{arrow} {esc(delta if isinstance(delta, str) else f"{delta:+.1f}")} 지난 보고서 대비</div>'
    small = " s" if len(str(value)) > 9 else ""
    return (f'<div class="kpi" title="{esc(tip)}"><div class="l">{esc(label)}</div>'
            f'<div class="v{small}">{esc(value)}</div>{d}</div>')


def table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()
    setup_style()
    run = os.path.abspath(args.run_dir)
    rdir = os.path.join(run, "reports")
    os.makedirs(rdir, exist_ok=True)
    hist_path = os.path.join(rdir, "kpi_history.json")
    hist = json.load(open(hist_path, encoding="utf-8")) if os.path.exists(hist_path) else []
    idx = args.index or (len(hist) + 1)

    def read(name):
        p = os.path.join(run, name)
        try:
            return pd.read_csv(p) if os.path.exists(p) else None
        except Exception:  # noqa: BLE001
            return None
    prog, eps, ev = read("progress.csv"), read("episodes.csv"), read("curriculum_events.csv")
    ctrl_ev = read("control_events.csv")
    if prog is not None and TS in prog:
        prog = prog.sort_values(TS)
    status = {}
    try:
        status = json.load(open(os.path.join(run, "status.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    stages = None
    try:
        import yaml
        stages = [s["name"] for s in yaml.safe_load(open(os.path.join(run, "config.yaml"), encoding="utf-8"))
                  ["curriculum"]["stages"]]
    except Exception:  # noqa: BLE001
        stages = [str(i) for i in range(8)]

    n_rows = 0 if eps is None else len(eps)
    older = [h for h in hist if h["index"] < idx]
    prev = older[-1] if older else None
    start_row = prev["rows"] if prev and prev.get("rows", 0) <= n_rows else 0
    interval = eps.iloc[start_row:] if eps is not None else None
    total_eps = args.episodes if args.episodes is not None else n_rows
    steps = args.timesteps if args.timesteps is not None else (int(prog[TS].max()) if prog is not None and TS in prog else 0)

    def last(col):
        if prog is None or col not in prog:
            return None
        s = prog[col].dropna()
        return float(s.iloc[-1]) if len(s) else None
    cur_stage = int(status.get("stage", last("curriculum/stage") or 0))
    win_sr = status.get("success_rate_window", last("curriculum/success_rate_window"))
    iv = interval if interval is not None and len(interval) else None
    cur_iv = iv[iv["stage"] == cur_stage] if iv is not None else None
    if cur_iv is not None and len(cur_iv) == 0:
        cur_iv = iv
    k = {
        "index": idx, "time": time.strftime("%Y-%m-%d %H:%M"), "episodes": int(total_eps), "rows": int(n_rows),
        "timesteps": int(steps), "stage": cur_stage,
        "interval_n": 0 if iv is None else int(len(iv)),
        "interval_success": float(cur_iv["success"].mean()) if cur_iv is not None else float("nan"),
        "interval_fall": float(cur_iv["fell"].mean()) if cur_iv is not None else float("nan"),
        "interval_return": float(cur_iv["return"].mean()) if cur_iv is not None else float("nan"),
        "interval_speed": float(cur_iv["mean_speed"].mean()) if cur_iv is not None else float("nan"),
        "stand_time": float(cur_iv.loc[cur_iv["stand_time"] >= 0, "stand_time"].mean())
        if cur_iv is not None and (cur_iv["stand_time"] >= 0).any() else float("nan"),
        "window_success": None if win_sr is None else float(win_sr),
        "fps": float(status.get("fps") or last("time/fps") or 0),
    }

    def dlt(key, scale=1.0):
        if not prev or prev.get(key) is None or k.get(key) is None:
            return None
        a, b = k[key], prev[key]
        if any(isinstance(v, float) and math.isnan(v) for v in (a, b)):
            return None
        return (a - b) * scale

    # ---- 3D 녹화 ----
    rec, scenes, top_img, ep_charts, rec_err = None, [], None, {}, None
    if args.replay and os.path.exists(args.replay):
        from spider_rl.replay import load_replay
        rec = load_replay(args.replay)
        try:
            scenes, top_img = render_scenes(rec)
        except Exception as ex:  # noqa: BLE001  (그래픽 환경 없음 등)
            rec_err = f"3D 렌더링 불가: {ex}"
        try:
            ep_charts = episode_charts(rec)
        except Exception as ex:  # noqa: BLE001
            rec_err = (rec_err or "") + f" / 거동 그래프 실패: {ex}"

    charts = training_charts(prog, ev, eps)
    hist_new = older + [k]
    trend = trend_chart(hist_new)

    # ---- 요약 문장 ----
    stage_name = stages[cur_stage] if cur_stage < len(stages) else str(cur_stage)
    lines = [f"이번 구간 <b>{k['interval_n']:,}판</b> (누적 {k['episodes']:,}판, {k['timesteps'] / 1e6:.1f}M 스텝). "
             f"현재 단계는 <b>{cur_stage}: {esc(stage_name)}</b>입니다."]
    if not math.isnan(k["interval_success"]):
        d = dlt("interval_success", 100)
        lines.append(f"현재 단계 구간 성공률은 <b>{pct(k['interval_success'])}</b>"
                     + (f" (지난 보고서 대비 {d:+.1f}%p)" if d is not None else "") + ", "
                     f"뒤집힘 {pct(k['interval_fall'])}.")
    try:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(run, "config.yaml"), encoding="utf-8"))
        need = cfg["curriculum"]["stages"][cur_stage].get("promote_success")
        if need and need <= 1 and k["window_success"] is not None:
            gap = need - k["window_success"]
            lines.append(f"승급 기준 {need * 100:.0f}%까지 " + (f"<b>{gap * 100:.1f}%p</b> 남았습니다." if gap > 0
                                                         else "도달했습니다 (곧 다음 단계로 승급)."))
    except Exception:  # noqa: BLE001
        pass
    if prev and prev["stage"] != cur_stage:
        lines.append(f"지난 보고서 이후 <b>{prev['stage']} → {cur_stage}단계로 승급</b>했습니다.")
    if rec is not None:
        st = rec["stats"]
        res = "성공" if st.get("success") else ("뒤집힘" if st.get("fell") else "시간 초과")
        lines.append(f"녹화한 대표 에피소드 결과: <b>{res}</b> (목표 {st.get('init_dist', 0):.2f} m → "
                     f"남은 거리 {st.get('final_dist', 0):.2f} m, {st.get('ep_time', 0):.1f}초).")

    # ---- 표 ----
    tbl_interval = ""
    if iv is not None:
        g = iv.groupby("stage").agg(판수=("success", "size"), 성공률=("success", "mean"), 뒤집힘=("fell", "mean"),
                                    시간초과=("timeout", "mean"), 평균보상=("return", "mean"),
                                    평균남은거리=("final_dist", "mean"), 평균에너지J=("energy", "mean")).reset_index()
        g.insert(0, "단계", [f"{int(s)}: {stages[int(s)] if int(s) < len(stages) else s}" for s in g["stage"]])
        g = g.drop(columns=["stage"])
        for c in ["성공률", "뒤집힘", "시간초과"]:
            g[c] = g[c].map(lambda v: f"{v * 100:.1f}%")
        for c in ["평균보상", "평균남은거리", "평균에너지J"]:
            g[c] = g[c].map(lambda v: f"{v:.2f}")
        tbl_interval = table(g)
    tbl_hist = table(pd.DataFrame([{
        "#": h["index"], "시각": h["time"], "누적 판수": f"{h['episodes']:,}", "스텝(M)": f"{h['timesteps'] / 1e6:.1f}",
        "단계": h["stage"], "구간 판수": f"{h['interval_n']:,}", "구간 성공률": pct(h["interval_success"]),
        "뒤집힘": pct(h["interval_fall"]), "평균 보상": num(h["interval_return"]),
        "속도(m/s)": num(h["interval_speed"], "{:.3f}")} for h in reversed(hist_new[-12:])]))
    tbl_ev = ""
    if ev is not None and len(ev):
        e2 = ev.copy()
        e2["timesteps"] = e2["timesteps"].map(lambda v: f"{int(v):,}")
        e2["success_rate"] = e2["success_rate"].map(lambda v: "" if pd.isna(v) else f"{float(v) * 100:.1f}%")
        e2["wall_time"] = e2["wall_time"].map(lambda v: _fmt_dur(v) if v else "-")
        e2.columns = ["스텝", "경과", "이벤트", "단계", "단계 이름", "성공률"]
        e2["이벤트"] = e2["이벤트"].map({"start": "시작", "promote": "승급", "end": "종료"}).fillna(e2["이벤트"])
        tbl_ev = table(e2)
    tbl_ctrl = table(ctrl_ev.tail(15)) if ctrl_ev is not None and len(ctrl_ev) else ""

    # ---- HTML ----
    run_name = os.path.basename(run)
    tiles = "".join([
        kpi("현재 단계", f"{cur_stage}: {stage_name}", tip="커리큘럼 단계"),
        kpi("구간 성공률", pct(k["interval_success"]), dlt("interval_success", 100), True, "이번 구간, 현재 단계 에피소드 기준"),
        kpi("승급 판단 성공률", pct(k["window_success"]), dlt("window_success", 100), True, "최근 300판 (승급 기준과 비교)"),
        kpi("뒤집힘 비율", pct(k["interval_fall"]), dlt("interval_fall", 100), False, "뒤집혀 실패한 비율"),
        kpi("평균 보상", num(k["interval_return"], "{:.1f}"), dlt("interval_return"), True),
        kpi("목표 방향 속도", num(k["interval_speed"], "{:.3f} m/s"), None, True),
        kpi("일어서기 시간", num(k["stand_time"], "{:.2f} s"), None, False, "엎드린 자세에서 서기 성공까지"),
        kpi("학습 속도", f"{k['fps']:,.0f} 스텝/s", None, True),
    ])

    def fig(src, cap):
        return f'<figure><img class="chart" src="{src}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>' if src else ""

    sec_3d = ""
    if rec is not None:
        m = rec["meta"]
        rel = os.path.relpath(args.replay, os.path.dirname(os.path.dirname(run))) if args.replay else ""
        imgs = "".join(fig(src, cap) for cap, src in scenes)
        sec_3d = f"""<h2>3D 장면: 현재 정책 대표 에피소드</h2>
<p class="sub">{m['stage']}단계 {esc(m['stage_name'])} · 시드 {m['seed']} · 결정적 정책으로 녹화.
{esc(rec_err) if rec_err else ''}</p>
<div class="grid4">{imgs}</div>
<div class="grid2" style="margin-top:12px">{fig(top_img, '위에서 본 경기장과 이동 궤적 (파란 선)')}
<div>{fig(ep_charts.get('gait'), '보행 패턴')}{fig(ep_charts.get('height_goal'), '몸체 높이 · 목표까지 거리')}</div></div>
<div class="grid2" style="margin-top:12px">{fig(ep_charts.get('joints'), '관절 궤적')}{fig(ep_charts.get('rewards'), '보상 항목 누적')}</div>
<p class="sub" style="margin-top:10px">3D로 다시 보기: <code>python scripts/viewer3d.py replay "{esc(rel)}"</code><br>
같은 조건으로 재현: <code>python scripts/viewer3d.py reproduce "{esc(rel)}"</code></p>"""

    body = f"""<div class="wrap">
<h1>학습 보고서 #{idx}</h1>
<div class="sub">{esc(run_name)} · {k['time']} 생성 · 누적 {k['episodes']:,}판 · {k['timesteps']:,} 스텝
 · 경과 {esc(_fmt_dur(status.get('elapsed_sec')))} · <a href="index.html">전체 보고서 목록</a></div>
<div class="summary">{' '.join(lines)}</div>
<div class="kpis">{tiles}</div>
{sec_3d}
<h2>학습 곡선</h2>{fig(charts.get('curves'), '성공률 · 보상 · 종료 원인 · 커리큘럼 단계 (점선 = 단계 승급)')}
<h2>보상 분해</h2>{fig(charts.get('rewards'), '어떤 보상/패널티가 학습을 이끄는지')}
<h2>이번 구간 단계별 결과</h2>{tbl_interval or '<p class="sub">이번 구간 에피소드 없음</p>'}
<h2>보고서별 추이</h2>{fig(trend, '보고서마다의 구간 성공률·뒤집힘·단계') if trend else ''}{tbl_hist}
<div class="grid2" style="margin-top:12px">{fig(charts.get('heatmap'), '단계 × 목표 거리 성공률')}{fig(charts.get('ppo'), 'PPO 학습 지표')}</div>
<h2>커리큘럼 이벤트</h2>{tbl_ev or '<p class="sub">아직 없음</p>'}
<h2>학습 중 설정 변경</h2>{tbl_ctrl or '<p class="sub">변경 없음</p>'}
</div>"""
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>학습 보고서 #{idx}</title><style>{CSS}</style>'
            f'</head><body>{body}</body></html>')
    name = f"report_{idx:04d}_ep{k['episodes']}.html"
    with open(os.path.join(rdir, name), "w", encoding="utf-8") as f:
        f.write(page)
    k["file"] = name
    hist_new[-1] = k
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist_new, f, ensure_ascii=False, indent=1)
    write_index(rdir, run_name, hist_new, trend)
    print(f"[보고서] {os.path.join(rdir, name)}")


def _fmt_dur(sec):
    if not sec:
        return "-"
    h, r = divmod(int(float(sec)), 3600)
    return f"{h}시간 {r // 60:02d}분" if h else f"{r // 60}분 {r % 60:02d}초"


def write_index(rdir, run_name, hist, trend):
    rows = "".join(
        f"<tr><td><a href=\"{esc(h.get('file', ''))}\">#{h['index']}</a></td><td>{esc(h['time'])}</td>"
        f"<td>{h['episodes']:,}</td><td>{h['timesteps'] / 1e6:.1f}</td><td>{h['stage']}</td>"
        f"<td>{pct(h['interval_success'])}</td><td>{pct(h['interval_fall'])}</td><td>{num(h['interval_return'])}</td></tr>"
        for h in reversed(hist))
    tr = f'<figure><img class="chart" src="{trend}"><figcaption>보고서별 추이</figcaption></figure>' if trend else ""
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>보고서 목록</title><meta http-equiv="refresh" content="120">'
            f'<style>{CSS}</style></head><body><div class="wrap"><h1>학습 보고서 목록</h1>'
            f'<div class="sub">{esc(run_name)} · 2분마다 자동 새로고침</div>{tr}<h2>보고서</h2>'
            f'<div class="tablewrap"><table><thead><tr><th>#</th><th>시각</th><th>누적 판수</th><th>스텝(M)</th>'
            f'<th>단계</th><th>구간 성공률</th><th>뒤집힘</th><th>평균 보상</th></tr></thead><tbody>{rows}</tbody>'
            f'</table></div></div></body></html>')
    with open(os.path.join(rdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
