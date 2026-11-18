"""보고서용 matplotlib 공통 스타일·유틸리티.

- 범주형 색상은 고정 순서로 사용 (색맹 검증된 8색 팔레트)
- 이중 y축 금지: 단위가 다른 지표는 별도 패널로
- 얇은 선(2px), 흐린 격자, 2개 이상 계열은 범례
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_CMAP = "Blues"

_KO = None


def setup_style() -> bool:
    """한글 폰트를 찾아 설정. 찾으면 True."""
    global _KO
    if _KO is not None:
        return _KO
    names = {f.name for f in font_manager.fontManager.ttflist}
    ko = None
    for cand in ("Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic", "Noto Sans CJK KR",
                 "Noto Sans KR", "UnDotum"):
        if cand in names:
            ko = cand
            break
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False, "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 110,
        "axes.unicode_minus": False,
    })
    if ko:
        plt.rcParams["font.family"] = ko
    _KO = ko is not None
    return _KO


def L(ko: str, en: str) -> str:
    """한글 폰트가 있으면 한글, 없으면 영문 라벨."""
    return ko if setup_style() else en


def savefig(fig, out_dir: str, name: str, saved: list | None = None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if saved is not None:
        saved.append(path)
    return path


def smooth(y, k: int):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if k <= 1 or len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k - 1, 0), mode="edge")
    return np.convolve(pad, ker, mode="valid")
