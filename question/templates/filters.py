"""
question/templates/filters.py
Jinja2 filters — 统一的数字/百分比格式化. 全部对 None/异常输入鲁棒 (返回 "N/A").
"""
from __future__ import annotations
from typing import Any


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def money(val: Any, *, decimals: int | None = None) -> str:
    """金额格式化: 12_345_678_901 -> '$12.3B', 840_000_000 -> '$840M'.

    - T / B: 1 位小数 (可被 decimals 覆盖)
    - M / K: 默认 0 位小数
    - < 1K:  2 位小数 (股价场景)
    - 负数加前缀 '-'
    - None / 非数: 'N/A'
    """
    f = _to_float(val)
    if f is None:
        return "N/A"
    sign = "-" if f < 0 else ""
    v = abs(f)
    if v >= 1e12:
        d = 1 if decimals is None else decimals
        return f"{sign}${v/1e12:.{d}f}T"
    if v >= 1e9:
        d = 1 if decimals is None else decimals
        return f"{sign}${v/1e9:.{d}f}B"
    if v >= 1e6:
        d = 0 if decimals is None else decimals
        return f"{sign}${v/1e6:.{d}f}M"
    if v >= 1e3:
        d = 0 if decimals is None else decimals
        return f"{sign}${v/1e3:.{d}f}K"
    d = 2 if decimals is None else decimals
    return f"{sign}${v:,.{d}f}"


def pct(val: Any, *, decimals: int = 2, signed: bool = True) -> str:
    """百分比格式化. 输入 0.1537 -> '+15.37%'.

    约定: 传入的是 *小数形式* (0.15 = 15%). 若已经是 15 这种大数字, 请自行 /100.
    """
    f = _to_float(val)
    if f is None:
        return "N/A"
    pct_val = f * 100
    if signed:
        return f"{pct_val:+.{decimals}f}%"
    return f"{pct_val:.{decimals}f}%"


def compact(val: Any, *, decimals: int = 0) -> str:
    """纯数量 (非货币) 的紧凑表示: 1_234_567 -> '1.2M'."""
    f = _to_float(val)
    if f is None:
        return "N/A"
    sign = "-" if f < 0 else ""
    v = abs(f)
    if v >= 1e9:
        return f"{sign}{v/1e9:.{max(decimals,1)}f}B"
    if v >= 1e6:
        return f"{sign}{v/1e6:.{max(decimals,1)}f}M"
    if v >= 1e3:
        return f"{sign}{v/1e3:.{max(decimals,1)}f}K"
    return f"{sign}{v:,.{decimals}f}"


# 便于 engine.register_filters 调用的集中导出
ALL_FILTERS = {
    "money": money,
    "pct": pct,
    "compact": compact,
}
