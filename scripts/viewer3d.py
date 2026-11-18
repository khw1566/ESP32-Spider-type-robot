"""3D 시각화: 정책 실시간 관찰 / 녹화 재생 / 결과 재현 / 영상 내보내기.

  # 학습된 정책을 3D로 실시간 관찰 (에피소드마다 녹화 파일 자동 저장)
  python scripts/viewer3d.py live --checkpoint runs/<실행>/checkpoints/latest.zip --stage 5
  python scripts/viewer3d.py live --policy stand --stage 0          # 기준선(스크립트) 정책
  # 녹화 파일 3D 재생 (여러 개 주면 PageUp/PageDown 으로 넘김)
  python scripts/viewer3d.py replay runs/<실행>/replays/*.npz
  # 결과 재현: 같은 설정·시드·정책으로 다시 시뮬레이션해 원본과 비교한 뒤 재생
  python scripts/viewer3d.py reproduce runs/<실행>/replays/xxx.npz
  # 녹화 파일 → mp4 (보고서·발표용)
  python scripts/viewer3d.py video runs/<실행>/replays/xxx.npz --camera track --out ep.mp4

3D 창 조작 (MuJoCo 기본: 마우스 왼쪽 드래그 회전, 오른쪽 드래그 이동, 휠 확대)
  Space      일시정지 / 재개
  ← / →      (재생) 1초 뒤로 / 앞으로,  일시정지 중에는 한 프레임씩
  ↑ / ↓      재생 속도 2배 / 절반 (0.125x ~ 8x)
  Home       처음부터 다시 (live: 같은 시드로 다시)
  PgUp/PgDn  이전 / 다음 녹화 파일 (live: 다음 에피소드, 새 시드)
  Insert     카메라 전환: 로봇 추적 → 자유 시점 → 위에서 내려다보기
  End        거리센서 광선·이동 궤적 표시 켜기/끄기
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from spider_rl.env import SpiderEnv  # noqa: E402
from spider_rl.replay import (apply_layout, config_for, list_replays, load_model_policy, load_replay,  # noqa: E402
                              replay_name, reproduce, run_dir_of, save_replay, scripted_policy,
                              sensor_segments)

K_SPACE, K_RIGHT, K_LEFT, K_DOWN, K_UP = 32, 262, 263, 264, 265
K_PGUP, K_PGDN, K_HOME, K_END, K_INSERT = 266, 267, 268, 269, 260
RAY_RGBA = [np.array([1.0, 0.85, 0.1, 0.9]), np.array([1.0, 0.5, 0.1, 0.9]),
            np.array([1.0, 0.5, 0.1, 0.9]), np.array([0.9, 0.2, 0.9, 0.9])]
TRAIL_RGBA = np.array([0.16, 0.47, 0.84, 0.9])


class Keys:
    """뷰어 스레드의 키 입력을 메인 루프로 전달."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[int] = []
        self.paused = False
        self.speed = 1.0
        self.overlay = True
        self.cam_mode = 0

    def __call__(self, key):
        with self.lock:
            if key == K_SPACE:
                self.paused = not self.paused
            elif key == K_UP:
                self.speed = min(8.0, self.speed * 2)
            elif key == K_DOWN:
                self.speed = max(0.125, self.speed / 2)
            elif key == K_END:
                self.overlay = not self.overlay
            elif key == K_INSERT:
                self.cam_mode = (self.cam_mode + 1) % 3
                self.events.append(key)
            else:
                self.events.append(key)

    def pop(self):
        with self.lock:
            ev, self.events = self.events, []
            return ev


def set_camera(viewer, model, mode):
    cam = viewer.cam
    if mode == 0:      # 로봇 추적
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("torso").id
        cam.distance, cam.elevation = 1.3, -28
    elif mode == 1:    # 자유 시점 (현재 위치에서 마우스로 조작)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:              # 경기장 전체 위에서
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0]
        cam.distance, cam.elevation, cam.azimuth = 7.5, -89, 90


def draw_overlay(viewer, model, data, cfg, dists, trail, show):
    scn = viewer.user_scn
    scn.ngeom = 0
    if not show:
        return
    if dists is not None:
        for (a, b), rgba in zip(sensor_segments(model, data, cfg, dists), RAY_RGBA):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, a, b)
            scn.ngeom += 1
    pts = trail[-400::4]
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), TRAIL_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, a, b)
        scn.ngeom += 1


def hud(viewer, left: str, right: str, help_text: str):
    try:
        viewer.set_texts([
            (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, ""),
        ])
    except Exception:  # noqa: BLE001  (구버전 mujoco)
        pass


HELP = ("Space pause  <-/-> seek/step  Up/Down speed  Home restart  "
        "PgUp/PgDn prev/next  Insert camera  End rays/trail")


def stats_text(st: dict) -> str:
    if not st:
        return ""
    res = "SUCCESS" if st.get("success") else ("FELL" if st.get("fell") else ("TIMEOUT" if st.get("timeout") else "END"))
    return f"{res}  return {st.get('return', 0):.1f}  final dist {st.get('final_dist', 0):.2f} m"


# ======================================================================
# 녹화 재생
# ======================================================================
def play_replays(files: list[str], start_paused=False):
    import mujoco.viewer
    idx = 0
    rec = load_replay(files[idx])
    env = SpiderEnv(rec["config"], record=False)
    m, d = env.model, env.data
    keys = Keys()
    keys.paused = start_paused

    def load(i):
        nonlocal rec
        r = load_replay(files[i])
        if r["config"] != rec["config"] or r["qpos"].shape[1] != m.nq:
            raise RuntimeError("설정이 다른 녹화 파일은 한 창에서 넘길 수 없습니다. 따로 여세요.")
        rec = r
        apply_layout(m, rec["layout"])

    apply_layout(m, rec["layout"])
    with mujoco.viewer.launch_passive(m, d, key_callback=keys) as viewer:
        set_camera(viewer, m, 0)
        frame, acc, last = 0, 0.0, time.time()
        while viewer.is_running():
            now = time.time()
            dt_wall, last = now - last, now
            T = len(rec["qpos"])
            fps = 1.0 / rec["meta"]["dt"]
            for ev in keys.pop():
                if ev == K_RIGHT:
                    frame = min(T - 1, frame + (1 if keys.paused else int(fps)))
                elif ev == K_LEFT:
                    frame = max(0, frame - (1 if keys.paused else int(fps)))
                elif ev == K_HOME:
                    frame = 0
                elif ev in (K_PGDN, K_PGUP) and len(files) > 1:
                    idx = (idx + (1 if ev == K_PGDN else -1)) % len(files)
                    try:
                        load(idx)
                    except RuntimeError as ex:
                        print(ex)
                    frame = 0
                elif ev == K_INSERT:
                    set_camera(viewer, m, keys.cam_mode)
            if not keys.paused and frame < T - 1:
                acc += dt_wall * keys.speed * fps
                step = int(acc)
                acc -= step
                frame = min(T - 1, frame + step)
            with viewer.lock():
                d.qpos[:] = rec["qpos"][frame]
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                k = max(0, frame - 1)
                ser = rec["series"]
                dists = ser["dists"][k] if len(ser.get("dists", [])) else None
                trail = [q[:3] for q in rec["qpos"][: frame + 1]]
                draw_overlay(viewer, m, d, rec["config"], dists, trail, keys.overlay)
            meta = rec["meta"]
            gd = ser["goal_dist"][k] if len(ser.get("goal_dist", [])) else 0.0
            left = (f"[{idx + 1}/{len(files)}] {os.path.basename(files[idx])}\n"
                    f"stage {meta['stage']} {meta['stage_name']}  seed {meta['seed']}\n"
                    f"t {frame / fps:5.2f}/{(T - 1) / fps:.2f} s   speed x{keys.speed:g}"
                    f"{'   PAUSED' if keys.paused else ''}\n"
                    f"goal dist {gd:.2f} m\n{stats_text(rec['stats']) if frame == T - 1 else ''}")
            hud(viewer, left, "", HELP)
            viewer.sync()
            time.sleep(1 / 60)


# ======================================================================
# 실시간 관찰 (정책 실행 + 녹화)
# ======================================================================
def live(args):
    import mujoco.viewer
    cfg = config_for(args.checkpoint)
    for s in args.set:
        from spider_rl.config import _set_dotted
        import yaml
        k, _, v = s.partition("=")
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    stage = args.stage
    if stage is None and args.checkpoint:
        import json
        cj = args.checkpoint.replace(".zip", "_curriculum.json")
        stage = json.load(open(cj, encoding="utf-8"))["stage"] if os.path.exists(cj) else 0
    stage = stage or 0
    env = SpiderEnv(cfg, stage=stage, progress=args.progress, record=True, fixed_stage=True)
    if args.checkpoint:
        policy = load_model_policy(args.checkpoint, cfg)
        save_dir = os.path.join(run_dir_of(args.checkpoint), "replays")
        meta = {"policy": "model", "checkpoint": os.path.abspath(args.checkpoint)}
    else:
        policy = scripted_policy(args.policy, env)
        save_dir = os.path.join("runs", "_replays")
        meta = {"policy": args.policy}
    keys = Keys()
    seed = args.seed

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys) as viewer:
        set_camera(viewer, env.model, 0)
        while viewer.is_running():
            # 한 에피소드 실행: 정책·물리 계산은 30 Hz 실시간(×속도)으로
            if hasattr(policy, "reset"):
                policy.reset()
            opts = {"stage": stage, "progress": args.progress}
            obs, _ = env.reset(seed=seed, options=opts)
            env.action_space.seed(seed)
            q0 = env.data.qpos.copy()
            from spider_rl.replay import snapshot_layout
            layout = snapshot_layout(env.model)
            lay = env.arena.layout
            done, restart, nxt = False, False, False
            info, trail, ret = {}, [env.data.qpos[:3].copy()], 0.0
            t_next = time.time()
            while viewer.is_running() and not done:
                for ev in keys.pop():
                    if ev == K_HOME:
                        restart = True
                    elif ev in (K_PGDN, K_RIGHT):
                        nxt = True
                    elif ev == K_INSERT:
                        set_camera(viewer, env.model, keys.cam_mode)
                if restart or nxt:
                    break
                if keys.paused:
                    viewer.sync()
                    time.sleep(1 / 30)
                    t_next = time.time()
                    continue
                with viewer.lock():
                    a = policy(obs)
                    obs, r, te, tr, info = env.step(a)
                    ret += r
                    done = te or tr
                    trail.append(env.data.qpos[:3].copy())
                    draw_overlay(viewer, env.model, env.data, cfg, env.dist_queue[0], trail, keys.overlay)
                gd = env.trace[-1]["goal_dist"] if env.trace else 0.0
                left = (f"LIVE  {'model' if args.checkpoint else args.policy}\n"
                        f"stage {stage} {env.stages[stage]['name']}  seed {seed}\n"
                        f"t {env.t * env.dt_ctrl:5.2f} s   speed x{keys.speed:g}\n"
                        f"goal dist {gd:.2f} m   return {ret:.1f}")
                hud(viewer, left, "", HELP)
                viewer.sync()
                t_next += env.dt_ctrl / keys.speed
                time.sleep(max(0.0, t_next - time.time()))
            if env.trace and (done or nxt) and not args.no_save:
                tr_ = env.trace
                rec = {"qpos": np.vstack([q0] + [s["qpos"] for s in tr_]), "layout": layout,
                       "series": {k: np.array([s[k] for s in tr_]) for k in
                                  ["t", "pos", "yaw", "up_z", "height", "q", "cmd", "target", "torque", "qvel",
                                   "contacts", "dists", "goal_dist", "energy", "action"]},
                       "rewards": {k: np.array([s["reward"].get(k, 0.0) for s in tr_]) for k in tr_[0]["reward"]},
                       "stats": info.get("ep_stats", {}), "start": np.asarray(lay.start, float),
                       "goal": None if lay.goal is None else np.asarray(lay.goal, float),
                       "meta": dict(meta, seed=int(seed), stage=int(stage), progress=float(args.progress),
                                    prone=None, dt=env.dt_ctrl, stage_name=env.stages[stage]["name"],
                                    recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"), complete=bool(done)),
                       "config": cfg}
                path = save_replay(rec, os.path.join(save_dir, replay_name(stage, seed, "live")))
                print(f"[녹화] {path}  {stats_text(rec['stats'])}")
            if done and viewer.is_running():
                hud(viewer, left + "\n" + stats_text(info.get("ep_stats", {})), "", HELP)
                end = time.time() + 1.5
                while time.time() < end and viewer.is_running():
                    viewer.sync()
                    time.sleep(1 / 30)
            if not restart:
                seed += 1


# ======================================================================
# 영상 내보내기
# ======================================================================
def export_video(path, out, camera="track", width=960, height=540, speed=1.0):
    import imageio
    rec = load_replay(path)
    env = SpiderEnv(rec["config"])
    m, d = env.model, env.data
    apply_layout(m, rec["layout"])
    renderer = mujoco.Renderer(m, height=height, width=width)
    fps = int(round(1 / rec["meta"]["dt"] * speed))
    cam = camera
    if camera == "top":
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0]
        cam.distance, cam.elevation, cam.azimuth = 7.0, -89, 90
    with imageio.get_writer(out, fps=fps) as w:
        for q in rec["qpos"]:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            w.append_data(renderer.render())
    renderer.close()
    print(f"[영상] {out}  ({len(rec['qpos'])} 프레임, {fps} fps)")
    return out


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += list_replays(p) or sorted(glob.glob(os.path.join(p, "*.npz")))
        else:
            out += sorted(glob.glob(p)) or [p]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("live", help="정책 3D 실시간 관찰 (+녹화)")
    a.add_argument("--checkpoint", default=None)
    a.add_argument("--policy", default="stand", choices=["stand", "zero", "random"])
    a.add_argument("--stage", type=int, default=None)
    a.add_argument("--progress", type=float, default=1.0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--set", action="append", default=[])
    a.add_argument("--no-save", action="store_true")
    b = sub.add_parser("replay", help="녹화 파일 3D 재생")
    b.add_argument("files", nargs="+", help="녹화 파일(.npz), 와일드카드, 또는 실행 폴더")
    b.add_argument("--paused", action="store_true")
    c = sub.add_parser("reproduce", help="같은 조건으로 재시뮬레이션해 결과 재현")
    c.add_argument("file")
    c.add_argument("--checkpoint", default=None, help="녹화에 기록된 체크포인트 대신 사용할 파일")
    c.add_argument("--no-view", action="store_true")
    v = sub.add_parser("video", help="녹화 파일 → mp4")
    v.add_argument("file")
    v.add_argument("--out", default=None)
    v.add_argument("--camera", default="track", choices=["track", "side", "top"])
    v.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    if args.mode == "live":
        live(args)
    elif args.mode == "replay":
        files = expand(args.files)
        if not files:
            sys.exit("녹화 파일이 없습니다.")
        play_replays(files, args.paused)
    elif args.mode == "reproduce":
        rec = load_replay(args.file)
        out = reproduce(rec, checkpoint=args.checkpoint)
        ok = out["max_qpos_error"] < 1e-6 and out["same_length"]
        print(f"[재현] 최대 자세 오차 {out['max_qpos_error']:.3g}, 길이 일치 {out['same_length']}, "
              f"성공 {out['orig_success']} → {out['new_success']}  →  {'완전 재현' if ok else '차이 있음'}")
        if not ok:
            print("  (다른 PC·라이브러리 버전에서는 부동소수점 차이로 조금씩 달라질 수 있습니다)")
        p = save_replay(out["rec"], args.file.replace(".npz", "_reproduced.npz"))
        print(f"[재현 녹화] {p}")
        if not args.no_view:
            play_replays([args.file, p])
    elif args.mode == "video":
        out = args.out or args.file.replace(".npz", f"_{args.camera}.mp4")
        export_video(args.file, out, args.camera, speed=args.speed)


if __name__ == "__main__":
    main()
