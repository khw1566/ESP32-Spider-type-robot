"""MuJoCo 뷰어로 모델/경기장 확인 (Windows에서 창이 뜸).

  python scripts/view_model.py                       # 엎드린 자세, 일어서기 스크립트 동작
  python scripts/view_model.py --stage 7             # 혼합 코스 배치 확인
  python scripts/view_model.py --checkpoint runs/<실행>/checkpoints/latest.zip --stage 3   # 정책 재생
  python scripts/view_model.py --save-xml robot.xml  # 생성된 MJCF 저장 (CAD 대조용)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from spider_rl.config import load_config  # noqa: E402
from spider_rl.env import SpiderEnv  # noqa: E402
from spider_rl.model_builder import build_mjcf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--save-xml", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config, None, args.set)
    if args.save_xml:
        with open(args.save_xml, "w", encoding="utf-8") as f:
            f.write(build_mjcf(cfg))
        print("저장:", args.save_xml)
        return
    import mujoco.viewer
    env = SpiderEnv(cfg, stage=args.stage, progress=1.0, fixed_stage=True)
    policy = None
    if args.checkpoint:
        sys.argv = sys.argv[:1]
        from evaluate import load_policy  # type: ignore
        policy, _ = load_policy(args.checkpoint, "model", env)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            i, done = 0, False
            while not done and viewer.is_running():
                t0 = time.time()
                if policy is not None:
                    a = policy(obs)
                else:
                    a = np.zeros(8)
                    a[1::2] = -0.6 + 1.2 * min(1.0, i / 30)   # 1초에 걸쳐 일어서기
                obs, r, te, tr, info = env.step(a)
                done = te or tr
                i += 1
                viewer.sync()
                time.sleep(max(0.0, env.dt_ctrl - (time.time() - t0)))
            if not viewer.is_running():
                break
            print(info.get("ep_stats", {}).get("success"))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
