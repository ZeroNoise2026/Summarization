"""
question/templates/functions.py
Jinja2 global functions — 后端确定性计算 (同比/环比/表格渲染), 避免 LLM 做算术.

⚠️ 输入约定: 所有 list[dict] 入参都视为 **最新在前** (date DESC).
   shared.db.get_earnings() / get_price_snapshots() 默认就是这个顺序.
"""
from __future__ import annotations
from typing import Any, Iterable
from .filters import money, pct


def _safe_get(row: dict, key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── 1. 同比 YoY ──────────────────────────────────────────────
def yoy(rows: list[dict], field: str) -> float | None:
    """同比 = (最新 - 4个季度前) / |4个季度前|. rows 最新在前, 至少 5 条.

    返回小数 (例 0.1537), 失败返回 None (模板里 None 会被 pct filter 渲染成 'N/A').
    """
    if not rows or len(rows) < 5:
        return None
    cur = _safe_get(rows[0], field)
    prev = _safe_get(rows[4], field)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


# ── 2. 环比 QoQ ──────────────────────────────────────────────
def qoq(rows: list[dict], field: str) -> float | None:
    """环比 = (最新 - 上个季度) / |上个季度|. rows 最新在前, 至少 2 条."""
    if not rows or len(rows) < 2:
        return None
    cur = _safe_get(rows[0], field)
    prev = _safe_get(rows[1], field)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


# ── 3. 财年合计 ────────────────────────────────────────────
def sum_fy(rows: list[dict], field: str, year: int | str) -> float | None:
    """按财年 (quarter 形如 "Q3 2025" 或 "2025Q3") 求和. 找不到任何该年的行返回 None."""
    year_s = str(year)
    total = 0.0
    found = False
    for r in rows or []:
        q = r.get("quarter") or r.get("period") or ""
        if year_s in str(q):
            v = _safe_get(r, field)
            if v is not None:
                total += v
                found = True
    return total if found else None


def sum_fy_info(rows: list[dict], field: str, year: int | str) -> dict:
    """同 sum_fy, 但额外返回实际聚合的季度数和季度标签 (用于 FY 是否完整的门控判断).

    Returns:
        {"value": float | None, "n_quarters": int, "quarters": [str,...]}

    调用方应:
      - 若 prior 年的 n_quarters 与 current 年不匹配 → 不显示 FY 对比 (避免部分年份冒充全年)
      - 若 n_quarters < 4 且用户问的是 FY → 标注为 YTD
    """
    import re as _re
    year_s = str(year)
    matched: list[tuple[int, dict]] = []
    for r in rows or []:
        q = str(r.get("quarter") or r.get("period") or "")
        if year_s not in q:
            continue
        m = _re.search(r"Q([1-4])", q, _re.IGNORECASE)
        if m:
            matched.append((int(m.group(1)), r))
    matched.sort(key=lambda x: x[0])

    total = 0.0
    found = False
    quarters: list[str] = []
    for q_num, r in matched:
        v = _safe_get(r, field)
        if v is not None:
            total += v
            found = True
            quarters.append(f"Q{q_num}")
    return {
        "value": total if found else None,
        "n_quarters": len(quarters),
        "quarters": quarters,
    }


# ── 3b. 指定季度的 YoY / QoQ (P0-A: 用户问 "Q3 2025" 时不再默认拿 rows[0]) ──
def _find_row_for_quarter(rows: list[dict], quarter_label: str) -> int | None:
    """在 rows 里找 quarter == 'Q3 2025' 的索引. 找不到返回 None."""
    target = str(quarter_label).upper().replace(" ", "")
    for i, r in enumerate(rows or []):
        q = str(r.get("quarter") or r.get("period") or "").upper().replace(" ", "")
        if q == target:
            return i
    return None


def yoy_at(rows: list[dict], field: str, quarter_label: str) -> float | None:
    """指定季度的 YoY = 该季度 - 去年同季度 / |去年同季度|.

    与 yoy() 不同: yoy() 硬编码用 rows[0] 与 rows[4], 假设数据按时间排序且连续.
    yoy_at() 按季度标签精确匹配, 避免用户问 Q3 2025 但 rows[0] 是 Q4 2025 的错误.
    """
    import re as _re
    i = _find_row_for_quarter(rows, quarter_label)
    if i is None:
        return None
    m = _re.match(r"Q([1-4])\s+(\d{4})", quarter_label.strip(), _re.IGNORECASE)
    if not m:
        return None
    q_num = m.group(1)
    prev_year = int(m.group(2)) - 1
    prev_label = f"Q{q_num} {prev_year}"
    j = _find_row_for_quarter(rows, prev_label)
    if j is None:
        return None
    cur = _safe_get(rows[i], field)
    prev = _safe_get(rows[j], field)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


def qoq_at(rows: list[dict], field: str, quarter_label: str) -> float | None:
    """指定季度的 QoQ. 上季度 = Q(n-1) 同年, 或 Q4 上一年 (若 n=1)."""
    import re as _re
    i = _find_row_for_quarter(rows, quarter_label)
    if i is None:
        return None
    m = _re.match(r"Q([1-4])\s+(\d{4})", quarter_label.strip(), _re.IGNORECASE)
    if not m:
        return None
    q_num = int(m.group(1))
    year = int(m.group(2))
    if q_num == 1:
        prev_label = f"Q4 {year - 1}"
    else:
        prev_label = f"Q{q_num - 1} {year}"
    j = _find_row_for_quarter(rows, prev_label)
    if j is None:
        return None
    cur = _safe_get(rows[i], field)
    prev = _safe_get(rows[j], field)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


# ── 4. 取最大/最小 ─────────────────────────────────────────
def pick_max(rows: list[dict], field: str) -> dict | None:
    best: dict | None = None
    best_v: float | None = None
    for r in rows or []:
        v = _safe_get(r, field)
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v = v
            best = r
    return best


# ── 5. 表格渲染 (后端预生成 markdown, LLM 侧仅 {{tables.xxx}}) ──
def render_table(
    rows: list[dict],
    columns: list[tuple[str, str, str]],
    *,
    limit: int | None = None,
) -> str:
    """把 list[dict] 渲染成 Markdown 表格.

    columns: [(display_name, field_name, formatter)] — formatter ∈
      "money" / "pct" / "compact" / "str" / "int" / "raw"

    示例:
        render_table(earnings, [
            ("Quarter", "quarter", "str"),
            ("EPS",     "eps",     "money_2"),
            ("Revenue", "revenue", "money"),
            ("Net Income", "net_income", "money"),
        ], limit=4)
    """
    if not rows:
        return "_No data available._"
    cols = columns
    use_rows = rows[:limit] if limit else rows

    def _fmt(val: Any, fm: str) -> str:
        if val is None:
            return "N/A"
        if fm == "money":
            return money(val)
        if fm == "money_2":  # EPS 专用: 保留 2 位
            try:
                return f"${float(val):,.2f}"
            except (TypeError, ValueError):
                return "N/A"
        if fm == "pct":
            return pct(val)
        if fm == "compact":
            from .filters import compact
            return compact(val)
        if fm == "int":
            try:
                return f"{int(float(val)):,}"
            except (TypeError, ValueError):
                return "N/A"
        if fm == "raw":
            return str(val)
        return str(val)

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body_lines = []
    for r in use_rows:
        cells = [_fmt(r.get(c[1]), c[2]) for c in cols]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body_lines)


ALL_FUNCTIONS = {
    "yoy": yoy,
    "qoq": qoq,
    "yoy_at": yoy_at,
    "qoq_at": qoq_at,
    "sum_fy": sum_fy,
    "sum_fy_info": sum_fy_info,
    "pick_max": pick_max,
    "render_table": render_table,
}
