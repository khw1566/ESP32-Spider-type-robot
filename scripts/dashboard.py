"""학습 관제 대시보드 (Python GUI, tkinter + matplotlib).

  python scripts/dashboard.py        (Windows: windows\\dashboard.bat)

탭 구성
  1) 학습 시작  : 프리셋·병렬 환경 수·시간 제한·시작 단계·이어받기 등을 정해 학습 실행
  2) 실시간 관제 : 진행 상태, 남은 시간·종료 예정 시각, 학습 중 시간 제한/엔트로피/학습률 변경, 지금 저장, 정지
  3) 그래프     : 성공률·보상·단계·엔트로피·보상 분해·FPS 실시간 그래프, 보고서용 그래프 일괄 생성
모든 입력 칸에 마우스를 올리면 설명(툴팁)이 나온다.
학습은 별도 프로세스라 대시보드를 닫아도 계속되고, 다시 열면 이어서 관제할 수 있다.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spider_rl import dashboard_backend as be  # noqa: E402
from spider_rl.config import ROOT, load_config  # noqa: E402
from spider_rl.plotting import PALETTE, setup_style  # noqa: E402

REFRESH_MS = 3000
GRAPH_REFRESH_MS = 15000
TS = "time/total_timesteps"


class Tooltip:
    """위젯에 마우스를 올리면 설명을 띄운다."""

    def __init__(self, widget, text: str):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", background="#fffbe6", relief="solid", borderwidth=1,
                 wraplength=380, padx=6, pady=4).pack()

    def hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def labeled(parent, row, label, widget, tip, col=0):
    lb = ttk.Label(parent, text=label)
    lb.grid(row=row, column=col, sticky="w", padx=(4, 8), pady=3)
    widget.grid(row=row, column=col + 1, sticky="we", pady=3)
    Tooltip(lb, tip)
    Tooltip(widget, tip)
    return widget


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        setup_style()
        self.title("ESP32 거미형 로봇 학습 관제")
        self.geometry("1180x780")
        self.minsize(980, 640)
        self.cfg = load_config()
        self.stage_names = [f"{i}: {s['name']}" for i, s in enumerate(self.cfg["curriculum"]["stages"])]
        self.procs: dict[str, subprocess.Popen] = {}
        self.current_run: str | None = None
        self._last_ctrl_run = None

        style = ttk.Style(self)
        try:
            style.theme_use("vista" if os.name == "nt" else "clam")
        except tk.TclError:
            pass
        style.configure("Big.TLabel", font=("", 13, "bold"))
        style.configure("Value.TLabel", font=("", 11))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_start = ttk.Frame(nb)
        self.tab_ctrl = ttk.Frame(nb)
        self.tab_graph = ttk.Frame(nb)
        self.tab_view = ttk.Frame(nb)
        nb.add(self.tab_start, text="  학습 시작  ")
        nb.add(self.tab_ctrl, text="  실시간 관제  ")
        nb.add(self.tab_graph, text="  그래프  ")
        nb.add(self.tab_view, text="  3D 보기 · 보고서  ")
        self.nb = nb
        self._build_start()
        self._build_ctrl()
        self._build_graph()
        self._build_view()
        self.refresh_runs()
        self.after(500, self.tick)
        self.after(1500, self.graph_tick)

    # ==================================================================
    # 1) 학습 시작
    # ==================================================================
    def _build_start(self):
        f = ttk.LabelFrame(self.tab_start, text="새 학습 / 이어서 학습")
        f.pack(fill="x", padx=10, pady=10)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self.v_preset = tk.StringVar(value="desktop")
        cb = ttk.Combobox(f, textvariable=self.v_preset, state="readonly",
                          values=["desktop", "school_pc", "quick_test", "(기본값만)"])
        labeled(f, 0, "PC 프리셋", cb, "configs 폴더의 덮어쓰기 설정.\n"
                "desktop: RTX 4070 SUPER 데스크탑 (병렬 12)\nschool_pc: 학교 PC (병렬 24, 7.5시간)\n"
                "quick_test: 설치 확인용 수 분짜리 실행")
        cb.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())

        self.v_tag = tk.StringVar(value="main")
        labeled(f, 0, "실행 이름 꼬리표", ttk.Entry(f, textvariable=self.v_tag),
                "실행 폴더 이름 끝에 붙는 메모. 예) main, ent005, link11cm", col=2)

        self.v_envs = tk.IntVar(value=12)
        labeled(f, 1, "병렬 환경 수", ttk.Spinbox(f, from_=1, to=64, textvariable=self.v_envs, width=8),
                "동시에 돌리는 시뮬레이션 개수. CPU 스레드 수 근처가 가장 빠름.\n"
                "데스크탑 실측: 8개 2,611 / 12개 3,417 / 16개 3,509 스텝/초")
        self.v_hours = tk.DoubleVar(value=7.5)
        labeled(f, 1, "시간 제한 (시간)", ttk.Spinbox(f, from_=0, to=200, increment=0.5, textvariable=self.v_hours,
                                                width=8),
                "이 시간이 지나면 체크포인트를 저장하고 자동 종료. 0 = 제한 없음.\n"
                "학습 중에도 '실시간 관제' 탭에서 바꿀 수 있음.", col=2)

        self.v_steps = tk.StringVar(value="200")
        labeled(f, 2, "총 학습 스텝 (백만)", ttk.Entry(f, textvariable=self.v_steps),
                "학습할 최대 스텝 수 (단위 백만). 시간 제한과 둘 중 먼저 도달하면 종료.\n"
                "이어서 학습할 때는 기존 스텝에 더해짐.")
        self.v_device = tk.StringVar(value="auto")
        labeled(f, 2, "연산 장치", ttk.Combobox(f, textvariable=self.v_device, state="readonly",
                                            values=["auto", "cpu", "cuda:0"]),
                "신경망 학습 장치. 작은 MLP 정책은 보통 CPU가 더 빠름(auto=CPU).\nGPU 비교 실험은 cuda:0", col=2)

        self.v_stage = tk.StringVar(value="(이어받기 상태 / 0단계)")
        labeled(f, 3, "시작 단계", ttk.Combobox(f, textvariable=self.v_stage, state="readonly",
                                            values=["(이어받기 상태 / 0단계)"] + self.stage_names),
                "커리큘럼 시작 단계. 기본값은 새 학습이면 0단계(일어서기),\n이어받기면 체크포인트에 저장된 단계.")
        self.v_fixed = tk.BooleanVar(value=False)
        labeled(f, 3, "단계 고정", ttk.Checkbutton(f, variable=self.v_fixed),
                "체크하면 승급 없이 시작 단계만 학습 (특정 단계 집중 학습·실험 비교용)", col=2)

        rf = ttk.Frame(f)
        self.v_resume = tk.StringVar(value="")
        ttk.Entry(rf, textvariable=self.v_resume).pack(side="left", fill="x", expand=True)
        ttk.Button(rf, text="찾아보기", command=self._browse_ckpt).pack(side="left", padx=4)
        ttk.Button(rf, text="최근 실행의 latest", command=self._latest_ckpt).pack(side="left")
        labeled(f, 4, "이어받을 체크포인트", rf,
                "비워 두면 처음부터 학습. 이전 실행의 checkpoints\\latest.zip 을 고르면\n"
                "모델·정규화 통계·커리큘럼 단계를 그대로 이어받음 (학교 PC 8시간 세션 이어가기).")
        rf.grid(columnspan=3)

        self.t_sets = tk.Text(f, height=4, width=60)
        labeled(f, 5, "추가 설정 (한 줄에 하나)", self.t_sets,
                "configs/default.yaml 값을 개별로 덮어씀. 형식: 키.키=값\n"
                "예) ppo.ent_coef=0.005\n    robot.link2_length=0.11\n    robot.motor_holding_torque=0.55")
        self.t_sets.grid(columnspan=3)

        self.v_report_every = tk.IntVar(value=int(self.cfg["train"].get("report_every_episodes", 5000)))
        labeled(f, 6, "보고서 주기 (판)", ttk.Spinbox(f, from_=0, to=1000000, increment=500,
                                               textvariable=self.v_report_every, width=10),
                "이 판(에피소드) 수마다 학습 보고서(HTML: 요약·KPI·차트·표·3D 장면)를 자동 생성하고\n"
                "대표 에피소드를 3D 녹화. 0 = 끔. 학습 중에도 '실시간 관제' 탭에서 바꿀 수 있음.\n"
                "참고: 초반 일어서기 단계는 판이 짧아(6초) 빨리 쌓이고, 후반은 판당 30~50초.")

        bf = ttk.Frame(self.tab_start)
        bf.pack(fill="x", padx=10)
        b = ttk.Button(bf, text="▶  학습 시작", command=self.start_training)
        b.pack(side="left", ipadx=12, ipady=4)
        Tooltip(b, "학습을 별도 프로세스로 실행. 대시보드를 닫아도 학습은 계속됨.")
        self.lbl_cmd = ttk.Label(bf, text="", foreground="#52514e", wraplength=900)
        self.lbl_cmd.pack(side="left", padx=12)

        info = ttk.LabelFrame(self.tab_start, text="안내")
        info.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(info, justify="left", text=(
            "• 학습 결과는 runs\\<날짜>_<PC이름>_<꼬리표>\\ 폴더에 저장됩니다 (로그, 체크포인트, 그래프).\n"
            "• 시간 제한에 도달하거나 '저장 후 정지'를 누르면 checkpoints\\latest.zip 을 저장하고 끝납니다.\n"
            "• 다음 세션은 '최근 실행의 latest' 버튼으로 이어받기를 채운 뒤 시작하면 됩니다.\n"
            "• PC가 절전 모드로 들어가면 학습이 멈추니 전원 설정의 절전을 꺼 두세요."
        )).pack(anchor="w", padx=8, pady=6)

    def _apply_preset(self):
        p = self.v_preset.get()
        if p == "desktop":
            self.v_envs.set(12)
            self.v_hours.set(7.5)
        elif p == "school_pc":
            self.v_envs.set(24)
            self.v_hours.set(7.5)
        elif p == "quick_test":
            self.v_envs.set(4)
            self.v_hours.set(0.2)
            self.v_steps.set("0.03")

    def _browse_ckpt(self):
        p = filedialog.askopenfilename(initialdir=be.RUNS_DIR, filetypes=[("체크포인트", "*.zip")])
        if p:
            self.v_resume.set(p)

    def _latest_ckpt(self):
        for name in be.list_runs():
            ck = be.latest_checkpoint(os.path.join(be.RUNS_DIR, name))
            if ck:
                self.v_resume.set(ck)
                return
        messagebox.showinfo("이어받기", "latest.zip 이 있는 실행 폴더가 없습니다.")

    def start_training(self):
        try:
            steps = float(self.v_steps.get()) * 1e6 if self.v_steps.get().strip() else None
            hours = float(self.v_hours.get())
            envs = int(self.v_envs.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("입력 오류", "숫자 칸(병렬 환경 수, 시간 제한, 학습 스텝)을 확인하세요.")
            return
        st = self.v_stage.get()
        stage = st.split(":")[0] if st and st[0].isdigit() else None
        preset = self.v_preset.get()
        params = {
            "run_name": be.new_run_name(self.v_tag.get().strip().replace(" ", "_")),
            "preset": None if preset.startswith("(") else preset,
            "n_envs": envs, "max_hours": hours, "timesteps": steps,
            "resume": self.v_resume.get().strip() or None, "stage": stage, "fixed_stage": self.v_fixed.get(),
            "device": self.v_device.get(),
            "sets": self.t_sets.get("1.0", "end").splitlines()
            + [f"train.report_every_episodes={int(self.v_report_every.get())}"],
        }
        if params["resume"] and not os.path.exists(params["resume"]):
            messagebox.showerror("이어받기", f"체크포인트 파일이 없습니다:\n{params['resume']}")
            return
        proc, run_dir = be.launch_training(params)
        self.procs[params["run_name"]] = proc
        self.lbl_cmd.config(text=f"실행됨: {params['run_name']}  (PID {proc.pid})")
        self.refresh_runs(select=params["run_name"])
        self.nb.select(self.tab_ctrl)

    # ==================================================================
    # 2) 실시간 관제
    # ==================================================================
    def _build_ctrl(self):
        top = ttk.Frame(self.tab_ctrl)
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(top, text="실행 폴더").pack(side="left")
        self.v_run = tk.StringVar()
        self.cb_run = ttk.Combobox(top, textvariable=self.v_run, state="readonly", width=60)
        self.cb_run.pack(side="left", padx=6)
        self.cb_run.bind("<<ComboboxSelected>>", lambda _e: self._select_run(self.v_run.get()))
        Tooltip(self.cb_run, "관제할 학습 실행. 최근 실행이 맨 위.")
        ttk.Button(top, text="목록 새로고침", command=self.refresh_runs).pack(side="left")

        body = ttk.Frame(self.tab_ctrl)
        body.pack(fill="x", padx=10, pady=4)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        sf = ttk.LabelFrame(body, text="진행 상태")
        sf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.status_vars = {}
        rows = [("state", "상태", "학습 중 / 정지 중 / 완료 / 시간 제한 종료 / 응답 없음"),
                ("stage", "커리큘럼 단계", "현재 학습 중인 단계와 단계 내 진행도(난이도 램프)"),
                ("success", "최근 성공률", "현재 단계의 최근 300 에피소드 성공률 (승급 기준과 비교)"),
                ("steps", "학습 스텝", "지금까지 학습한 스텝 / 목표 스텝"),
                ("fps", "속도", "초당 학습 스텝 (최근 20초 평균)"),
                ("elapsed", "경과 시간", "이번 실행 시작 후 경과 시간"),
                ("remaining", "남은 시간", "시간 제한까지 남은 시간"),
                ("end", "종료 예정", "시간 제한 기준 종료 예정 시각 (이 PC 시각)"),
                ("params", "엔트로피 / 학습률", "현재 적용 중인 PPO 엔트로피 계수와 학습률"),
                ("report", "보고서", "생성된 보고서 수와 다음 보고서 시점 (누적 판수 기준)")]
        for i, (k, lab, tip) in enumerate(rows):
            lb = ttk.Label(sf, text=lab)
            lb.grid(row=i, column=0, sticky="w", padx=8, pady=3)
            v = tk.StringVar(value="-")
            ttk.Label(sf, textvariable=v, style="Value.TLabel").grid(row=i, column=1, sticky="w", padx=8)
            Tooltip(lb, tip)
            self.status_vars[k] = v

        cf = ttk.LabelFrame(body, text="학습 중 설정 변경 (몇 초 안에 반영)")
        cf.grid(row=0, column=1, sticky="nsew")
        cf.columnconfigure(1, weight=1)
        self.v_live_hours = tk.DoubleVar(value=7.5)
        hf = ttk.Frame(cf)
        ttk.Spinbox(hf, from_=0, to=200, increment=0.5, textvariable=self.v_live_hours, width=8).pack(side="left")
        ttk.Button(hf, text="적용", command=self.apply_hours).pack(side="left", padx=4)
        ttk.Button(hf, text="+1시간", command=lambda: self.bump_hours(1)).pack(side="left")
        ttk.Button(hf, text="-1시간", command=lambda: self.bump_hours(-1)).pack(side="left", padx=4)
        labeled(cf, 0, "시간 제한 (시간)", hf,
                "학습 시작 기준 총 학습 시간. 지금 경과 시간보다 작게 하면 곧바로 저장 후 종료.\n0 = 제한 없음")

        self.v_ent = tk.StringVar()
        ef = ttk.Frame(cf)
        ttk.Entry(ef, textvariable=self.v_ent, width=12).pack(side="left")
        ttk.Button(ef, text="적용", command=lambda: self.apply_float("ent_coef", self.v_ent)).pack(side="left", padx=4)
        labeled(cf, 1, "엔트로피 계수", ef, "탐험 정도. 학습이 정체되면 올리고(예 0.005),\n행동이 너무 떨리면 내림(예 0.001).")

        self.v_lr = tk.StringVar()
        lf = ttk.Frame(cf)
        ttk.Entry(lf, textvariable=self.v_lr, width=12).pack(side="left")
        ttk.Button(lf, text="적용", command=lambda: self.apply_float("learning_rate", self.v_lr)).pack(side="left", padx=4)
        labeled(cf, 2, "학습률", lf, "PPO 학습률. 보상이 출렁이거나 approx KL이 크면 낮춤 (예 1e-4).")

        bf = ttk.Frame(cf)
        self.v_live_report = tk.StringVar()
        rf2 = ttk.Frame(cf)
        ttk.Entry(rf2, textvariable=self.v_live_report, width=12).pack(side="left")
        ttk.Button(rf2, text="적용", command=self.apply_report_every).pack(side="left", padx=4)
        ttk.Button(rf2, text="지금 보고서", command=self.report_now).pack(side="left")
        labeled(cf, 3, "보고서 주기 (판)", rf2, "N판마다 자동 보고서 + 3D 녹화. 0 = 끔.\n'지금 보고서'는 즉시 한 번 생성.")
        bf.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 4), padx=4)
        b1 = ttk.Button(bf, text="지금 저장", command=self.save_now)
        b1.pack(side="left")
        Tooltip(b1, "현재 모델을 checkpoints\\step_*.zip 과 latest.zip 으로 즉시 저장")
        b2 = ttk.Button(bf, text="■ 저장 후 정지", command=self.stop_run)
        b2.pack(side="left", padx=6)
        Tooltip(b2, "latest.zip 을 저장하고 학습을 정상 종료")
        b3 = ttk.Button(bf, text="결과 폴더 열기", command=self.open_run_dir)
        b3.pack(side="left")
        self.lbl_ctrl = ttk.Label(cf, text="", foreground="#52514e", wraplength=420)
        self.lbl_ctrl.grid(row=5, column=0, columnspan=2, sticky="w", padx=4)

        lf2 = ttk.LabelFrame(self.tab_ctrl, text="학습 로그 (최근)")
        lf2.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.t_log = tk.Text(lf2, height=12, wrap="none", font=("Consolas", 9))
        self.t_log.pack(fill="both", expand=True)

    def refresh_runs(self, select: str | None = None):
        runs = be.list_runs()
        self.cb_run["values"] = runs
        target = select or self.current_run or (runs[0] if runs else None)
        if target:
            self.v_run.set(target)
            self._select_run(target)

    def _select_run(self, name):
        self.current_run = name
        self._last_ctrl_run = None
        self.graph_tick(reschedule=False)
        if hasattr(self, "lb_reports"):
            self.refresh_view_lists()

    def run_dir(self):
        return os.path.join(be.RUNS_DIR, self.current_run) if self.current_run else None

    def _need_run(self):
        d = self.run_dir()
        if not d:
            messagebox.showinfo("관제", "실행 폴더를 먼저 선택하세요.")
            return None
        if not os.path.exists(os.path.join(d, "control.json")):
            messagebox.showwarning("관제", "이 실행은 아직 시작 중이거나, 관제 기능이 없는 이전 버전으로 실행되었습니다.")
            return None
        return d

    def apply_hours(self):
        d = self._need_run()
        if d:
            try:
                h = float(self.v_live_hours.get())
            except (ValueError, tk.TclError):
                return
            be.write_control(d, max_hours=h)
            self.lbl_ctrl.config(text=f"시간 제한 {h:g}시간 요청" if h > 0 else "시간 제한 해제 요청")

    def bump_hours(self, dh):
        try:
            self.v_live_hours.set(max(0.0, float(self.v_live_hours.get()) + dh))
        except (ValueError, tk.TclError):
            return
        self.apply_hours()

    def apply_float(self, key, var):
        d = self._need_run()
        if not d:
            return
        try:
            v = float(var.get())
        except ValueError:
            messagebox.showerror("입력 오류", "숫자를 입력하세요. 예) 0.002, 3e-4")
            return
        be.write_control(d, **{key: v})
        self.lbl_ctrl.config(text=f"{key} = {v:g} 요청")

    def apply_report_every(self):
        d = self._need_run()
        if not d:
            return
        try:
            v = int(float(self.v_live_report.get()))
        except ValueError:
            messagebox.showerror("입력 오류", "판 수(정수)를 입력하세요. 0 = 끔")
            return
        be.write_control(d, report_every_episodes=v)
        self.lbl_ctrl.config(text=f"보고서 주기 {v:,}판 요청" if v > 0 else "자동 보고서 끔 요청")

    def report_now(self):
        d = self._need_run()
        if d:
            be.write_control(d, report_now=True)
            self.lbl_ctrl.config(text="보고서 생성 요청됨 (녹화 후 1~2분 내 '3D 보기 · 보고서' 탭에 표시)")

    def save_now(self):
        d = self._need_run()
        if d:
            be.write_control(d, save_now=True)
            self.lbl_ctrl.config(text="저장 요청됨")

    def stop_run(self):
        d = self._need_run()
        if d and messagebox.askyesno("정지", "latest.zip 을 저장하고 학습을 종료할까요?"):
            be.write_control(d, stop=True)
            self.lbl_ctrl.config(text="정지 요청됨 (저장 후 종료)")

    def open_run_dir(self):
        d = self.run_dir()
        if d and os.path.isdir(d):
            if os.name == "nt":
                os.startfile(d)  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", d])

    def tick(self):
        try:
            self._update_status()
        finally:
            self.after(REFRESH_MS, self.tick)

    def _update_status(self):
        d = self.run_dir()
        if not d:
            return
        st = be.read_status(d)
        sv = self.status_vars
        if st is None:
            for v in sv.values():
                v.set("-")
            sv["state"].set("시작 중 또는 상태 정보 없음")
        else:
            names = {"running": "학습 중", "stopping": "정지 중", "finished": "완료", "time_limit": "시간 제한으로 종료",
                     "interrupted": "중단됨"}
            state = names.get(st.get("state"), st.get("state"))
            if st.get("stale"):
                state = "응답 없음 (프로세스가 종료되었을 수 있음)"
            sv["state"].set(state)
            sv["stage"].set(f"{st['stage']}: {st['stage_name']}  (진행도 {st['stage_progress'] * 100:.0f}%)")
            sr = st.get("success_rate_window")
            sv["success"].set(f"{sr * 100:.1f}%" if sr is not None else "-")
            tot = st.get("total_timesteps")
            sv["steps"].set(f"{st['timesteps'] / 1e6:.2f}M" + (f" / {tot / 1e6:.0f}M" if tot else ""))
            sv["fps"].set(f"{st['fps']:,.0f} 스텝/초" if st.get("fps") else "-")
            sv["elapsed"].set(be.fmt_duration(st["elapsed_sec"]))
            sv["remaining"].set(be.fmt_duration(st.get("remaining_sec")))
            end = st.get("end_time")
            sv["end"].set(time.strftime("%m/%d %H:%M", time.localtime(end)) if end else "제한 없음")
            sv["params"].set(f"{st['ent_coef']:g} / {st['learning_rate']:g}")
            nxt = st.get("next_report_at")
            sv["report"].set(f"{st.get('reports', 0)}개 생성 · 현재 {st.get('total_episodes', 0):,}판"
                             + (f" · 다음 {nxt:,}판" if nxt else " · 자동 보고서 꺼짐"))
            if self._last_ctrl_run != self.current_run:   # 실행을 바꾸면 입력칸을 현재 값으로 채움
                self._last_ctrl_run = self.current_run
                self.v_live_hours.set(st.get("max_hours", 0))
                self.v_ent.set(f"{st['ent_coef']:g}")
                self.v_lr.set(f"{st['learning_rate']:g}")
                self.v_live_report.set(str(st.get("report_every_episodes", 0)))
        log = be.tail(os.path.join(d, "train.log"), 30)
        self.t_log.delete("1.0", "end")
        self.t_log.insert("end", log)
        self.t_log.see("end")

    # ==================================================================
    # 3) 그래프
    # ==================================================================
    def _build_graph(self):
        bar = ttk.Frame(self.tab_graph)
        bar.pack(fill="x", padx=10, pady=(8, 0))
        b = ttk.Button(bar, text="보고서용 그래프 전체 생성", command=self.make_report)
        b.pack(side="left")
        Tooltip(b, "scripts/plot_report.py 로 30여 종 PNG 를 runs\\<실행>\\figures 에 저장하고 폴더를 엶")
        ttk.Button(bar, text="지금 새로고침", command=lambda: self.graph_tick(reschedule=False)).pack(side="left", padx=6)
        self.lbl_graph = ttk.Label(bar, text="", foreground="#52514e")
        self.lbl_graph.pack(side="left", padx=8)
        self.fig = Figure(figsize=(11, 6.4), dpi=90)
        self.axes = [self.fig.add_subplot(2, 3, i + 1) for i in range(6)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_graph)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def graph_tick(self, reschedule=True):
        try:
            self._draw_graphs()
        except Exception as ex:  # noqa: BLE001  (CSV 쓰는 도중 읽기 등)
            self.lbl_graph.config(text=f"그래프 갱신 대기: {ex.__class__.__name__}")
        if reschedule:
            self.after(GRAPH_REFRESH_MS, self.graph_tick)

    def _draw_graphs(self):
        d = self.run_dir()
        for ax in self.axes:
            ax.clear()
        path = os.path.join(d, "progress.csv") if d else None
        if not path or not os.path.exists(path):
            self.axes[0].set_title("progress.csv 없음 (학습 시작 직후에는 잠시 기다리세요)", fontsize=9)
            self.canvas.draw_idle()
            return
        df = pd.read_csv(path)
        panels = [("perf/success_rate", "성공률 (현재 단계)"), ("perf/ep_return_mean", "에피소드 보상"),
                  ("curriculum/stage", "커리큘럼 단계"), ("train/entropy_loss", "엔트로피 손실"),
                  (None, "보상 항목 (스텝당)"), ("time/fps", "학습 속도 (스텝/초)")]
        for ax, (col, title) in zip(self.axes, panels):
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)
            if col is None:
                cols = [c for c in df.columns if c.startswith("reward_terms_per_step/")]
                if cols:
                    sub = df[[TS] + cols].dropna()
                    top = sub[cols].abs().mean().sort_values(ascending=False).index[:6]
                    for i, c in enumerate(top):
                        ax.plot(sub[TS] / 1e6, sub[c], color=PALETTE[i], lw=1.2, label=c.split("/")[-1])
                    ax.legend(fontsize=6, ncol=2)
                continue
            if col in df:
                sub = df[[TS, col]].dropna()
                if col == "curriculum/stage":
                    ax.step(sub[TS] / 1e6, sub[col], where="post", color=PALETTE[0])
                    ax.set_ylim(-0.3, len(self.stage_names) - 0.7)
                    ax.set_yticks(range(len(self.stage_names)))
                else:
                    ax.plot(sub[TS] / 1e6, sub[col], color=PALETTE[0], lw=1.4)
        for ax in self.axes[3:]:
            ax.set_xlabel("백만 스텝", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.lbl_graph.config(text=f"{self.current_run}  ·  {time.strftime('%H:%M:%S')} 갱신")

    def make_report(self):
        d = self.run_dir()
        if not d:
            return
        self.lbl_graph.config(text="그래프 생성 중...")
        self.update_idletasks()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plot_report.py"), d],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            messagebox.showerror("그래프 생성 실패", r.stderr[-2000:])
            return
        self.lbl_graph.config(text=r.stdout.splitlines()[0] if r.stdout else "완료")
        fig_dir = os.path.join(d, "figures")
        if os.name == "nt" and os.path.isdir(fig_dir):
            os.startfile(fig_dir)  # noqa: S606


def _build_view(self):
    import webbrowser
    from spider_rl.replay import list_replays
    self._webbrowser = webbrowser
    self._list_replays = list_replays
    t = self.tab_view
    t.columnconfigure(0, weight=1)
    t.columnconfigure(1, weight=1)
    t.rowconfigure(0, weight=1)

    lf = ttk.LabelFrame(t, text="학습 보고서 (N판마다 자동 생성)")
    lf.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
    self.lb_reports = tk.Listbox(lf, height=14)
    self.lb_reports.pack(fill="both", expand=True, padx=6, pady=6)
    self.lb_reports.bind("<Double-Button-1>", lambda _e: self.open_report())
    bf = ttk.Frame(lf)
    bf.pack(fill="x", padx=6, pady=(0, 6))
    b = ttk.Button(bf, text="보고서 열기", command=self.open_report)
    b.pack(side="left")
    Tooltip(b, "선택한 보고서를 웹브라우저로 엶 (요약·KPI·3D 장면·차트·표가 한 파일에 들어 있음)")
    ttk.Button(bf, text="전체 목록·추이", command=lambda: self.open_report(index=True)).pack(side="left", padx=4)
    ttk.Button(bf, text="지금 보고서 만들기", command=self.report_now).pack(side="left")

    rf = ttk.LabelFrame(t, text="3D 녹화 (보고서·실시간 관찰 때 자동 저장)")
    rf.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    self.lb_replays = tk.Listbox(rf, height=14, selectmode="extended")
    self.lb_replays.pack(fill="both", expand=True, padx=6, pady=6)
    self.lb_replays.bind("<Double-Button-1>", lambda _e: self.play_replay())
    bf2 = ttk.Frame(rf)
    bf2.pack(fill="x", padx=6, pady=(0, 6))
    for text, cmd, tip in [("3D 재생", self.play_replay, "선택한 녹화를 3D 창에서 재생 (여러 개 선택 시 PgUp/PgDn 으로 넘김)\n"
                                                         "Space 일시정지, ←/→ 되감기·빨리감기, ↑/↓ 속도, Insert 카메라"),
                           ("결과 재현", self.reproduce_replay, "녹화에 저장된 설정·시드·정책으로 다시 시뮬레이션해\n"
                                                              "원본과 같은 결과가 나오는지 확인한 뒤 둘을 3D로 재생"),
                           ("영상 저장", self.video_replay, "선택한 녹화를 mp4 로 저장 (보고서·발표용)")]:
        bb = ttk.Button(bf2, text=text, command=cmd)
        bb.pack(side="left", padx=(0, 4))
        Tooltip(bb, tip)

    lv = ttk.LabelFrame(t, text="정책 3D 실시간 관찰")
    lv.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
    self.v_view_stage = tk.StringVar(value=self.stage_names[0])
    self.v_view_seed = tk.IntVar(value=0)
    self.v_view_src = tk.StringVar(value="선택한 실행의 latest 체크포인트")
    row = ttk.Frame(lv)
    row.pack(fill="x", padx=6, pady=6)
    cb = ttk.Combobox(row, textvariable=self.v_view_src, state="readonly", width=30,
                      values=["선택한 실행의 latest 체크포인트", "기준선: 스크립트 일어서기", "기준선: 무작위 행동"])
    cb.pack(side="left")
    Tooltip(cb, "관찰할 정책. 학습 중이어도 마지막 저장된 체크포인트로 볼 수 있음")
    ttk.Label(row, text="  단계").pack(side="left")
    c2 = ttk.Combobox(row, textvariable=self.v_view_stage, state="readonly", values=self.stage_names, width=18)
    c2.pack(side="left", padx=4)
    Tooltip(c2, "어떤 커리큘럼 단계 환경(장애물·단차 포함 여부 등)에서 볼지")
    ttk.Label(row, text="시드").pack(side="left")
    ttk.Spinbox(row, from_=0, to=99999, textvariable=self.v_view_seed, width=7).pack(side="left", padx=4)
    b = ttk.Button(row, text="3D로 보기", command=self.live_view)
    b.pack(side="left", padx=8)
    Tooltip(b, "MuJoCo 3D 창에서 정책을 실시간으로 실행. 판마다 녹화 파일이 자동 저장되어 나중에 재생·재현 가능.\n"
               "PgDn 다음 판, Home 같은 시드로 다시, Space 일시정지, End 센서 광선·궤적 표시")
    self.lbl_view = ttk.Label(lv, text="", foreground="#52514e")
    self.lbl_view.pack(anchor="w", padx=6, pady=(0, 6))
    self.after(2000, self.view_tick)


def _view_tick(self):
    try:
        self.refresh_view_lists()
    finally:
        self.after(10000, self.view_tick)


def _refresh_view_lists(self):
    d = self.run_dir()
    reps, plays = [], []
    if d:
        rd = os.path.join(d, "reports")
        if os.path.isdir(rd):
            reps = sorted([f for f in os.listdir(rd) if f.startswith("report_") and f.endswith(".html")], reverse=True)
        plays = self._list_replays(d)
    if reps != getattr(self, "_reps", None):
        self._reps = reps
        self.lb_reports.delete(0, "end")
        for f in reps:
            self.lb_reports.insert("end", f)
    if plays != getattr(self, "_plays", None):
        self._plays = plays
        self.lb_replays.delete(0, "end")
        for f in plays:
            self.lb_replays.insert("end", os.path.basename(f))


def _open_report(self, index=False):
    d = self.run_dir()
    if not d:
        return
    if index:
        p = os.path.join(d, "reports", "index.html")
    else:
        sel = self.lb_reports.curselection()
        if not sel and self._reps:
            sel = (0,)
        if not sel:
            messagebox.showinfo("보고서", "아직 보고서가 없습니다. '지금 보고서 만들기'를 누르거나 주기를 기다리세요.")
            return
        p = os.path.join(d, "reports", self._reps[sel[0]])
    if os.path.exists(p):
        self._webbrowser.open("file:///" + os.path.abspath(p).replace(os.sep, "/"))


def _selected_replays(self):
    sel = self.lb_replays.curselection()
    if not sel and self._plays:
        sel = (0,)
    return [self._plays[i] for i in sel]


def _viewer(self, args, wait_output=False, done=None):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "viewer3d.py")] + args
    if not wait_output:
        subprocess.Popen(cmd, cwd=ROOT)
        return
    import threading

    def work():
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.after(0, lambda: done(r))
    threading.Thread(target=work, daemon=True).start()


def _play_replay(self):
    files = self._selected_replays()
    if files:
        self._viewer(["replay"] + files)
        self.lbl_view.config(text=f"3D 재생: {len(files)}개 파일")


def _reproduce_replay(self):
    files = self._selected_replays()
    if not files:
        return
    f = files[0]
    self.lbl_view.config(text="재현 시뮬레이션 중...")

    def done(r):
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[재현")]
        msg = "\n".join(lines) or r.stderr[-1500:]
        self.lbl_view.config(text=lines[0] if lines else "재현 실패")
        messagebox.showinfo("결과 재현", msg)
        rp = f.replace(".npz", "_reproduced.npz")
        if os.path.exists(rp):
            self._viewer(["replay", f, rp])
    self._viewer(["reproduce", f, "--no-view"], wait_output=True, done=done)


def _video_replay(self):
    files = self._selected_replays()
    if not files:
        return
    self.lbl_view.config(text="영상 저장 중...")

    def done(r):
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[영상")]
        self.lbl_view.config(text=lines[-1] if lines else ("영상 저장 실패: " + r.stderr[-300:]))
    self._viewer(["video", files[0], "--camera", "track"], wait_output=True, done=done)


def _live_view(self):
    src = self.v_view_src.get()
    stage = self.v_view_stage.get().split(":")[0]
    args = ["live", "--stage", stage, "--seed", str(int(self.v_view_seed.get()))]
    if src.startswith("선택한"):
        d = self.run_dir()
        ck = be.latest_checkpoint(d) if d else None
        if not ck:
            messagebox.showinfo("3D 보기", "선택한 실행에 아직 체크포인트(latest.zip)가 없습니다.\n"
                                         "'실시간 관제' 탭에서 '지금 저장'을 누르거나 기준선 정책을 고르세요.")
            return
        args += ["--checkpoint", ck]
    else:
        args += ["--policy", "stand" if "일어서기" in src else "random"]
    self._viewer(args)
    self.lbl_view.config(text="3D 창을 여는 중... (창을 닫으면 종료)")


for _n, _f in [("_build_view", _build_view), ("view_tick", _view_tick), ("refresh_view_lists", _refresh_view_lists),
               ("open_report", _open_report), ("_selected_replays", _selected_replays), ("_viewer", _viewer),
               ("play_replay", _play_replay), ("reproduce_replay", _reproduce_replay),
               ("video_replay", _video_replay), ("live_view", _live_view)]:
    setattr(Dashboard, _n, _f)


def main():
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
