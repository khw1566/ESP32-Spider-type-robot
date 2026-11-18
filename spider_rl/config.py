"""설정 파일 로딩/병합 유틸리티."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "default.yaml")


def deep_update(base: dict, override: dict) -> dict:
    """override 의 값을 base 에 재귀적으로 덮어쓴다 (base 복사본 반환)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(path: str | None = None, overrides: list[str] | None = None,
                sets: list[str] | None = None) -> dict:
    """기본 설정 → override 파일들 → 'a.b.c=값' 형태 개별 설정 순으로 병합."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG):
        # 사용자가 부분 설정 파일만 넘긴 경우에도 기본값을 채운다
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            cfg = deep_update(yaml.safe_load(f), cfg)
    for ov in overrides or []:
        with open(ov, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    for s in sets or []:
        key, _, raw = s.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
