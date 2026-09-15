"""把报告里的任意一个指标"顶"成主指标,好让同一套配对检验去比不同的策略。

    # 比"只看 top-1"与"放宽到 top-3"两种检索策略:
    python eval/metric_view.py eval/report_definite.json \
        --field card_hit_top1 --gate expect_card -o eval/runs/def_top1.json
    python eval/metric_view.py eval/report_definite.json \
        --field card_hit_top3 --gate expect_card -o eval/runs/def_top3.json
    python eval/compare.py eval/runs/def_top1.json eval/runs/def_top3.json \
        --metrics ok --label-a "top-1" --label-b "top-3"

为什么需要 `--gate`,以及为什么不能省:

    检索类指标(`card_hit_top1` / `card_hit_top3`)对**没有期望卡片标注**的题
    一律记成 False —— 那表示"这道题不参与这个指标",**不是**"这一版失败了"。
    如果不加门禁直接把它们当主指标,分母会凭空变大,比率被稀释,
    配对检验也会把两边都算成错。

    所以 gate 字段为假的题,`ok` 置为 None,由 compare.py 的配对逻辑整对丢掉。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def make_view(report: dict, field: str, gate: str | None = None) -> tuple[dict, int]:
    """返回 (新的 report, 可比题数)。不改动原对象。"""
    out = dict(report)
    records = []
    comparable = 0
    for rec in report["records"]:
        new = dict(rec)
        keep = rec.get(field)
        if gate and not rec.get(gate):
            keep = None
        new["ok"] = None if keep is None else bool(keep)
        if new["ok"] is not None:
            comparable += 1
        records.append(new)
    out["records"] = records
    return out, comparable


def main() -> int:
    ap = argparse.ArgumentParser(description="把某个逐题指标顶成主指标 ok")
    ap.add_argument("report", help="run_eval.py 产出的报告 JSON")
    ap.add_argument("--field", required=True, help="要顶成 ok 的字段名")
    ap.add_argument("--gate", default=None,
                    help="门禁字段:该字段为假的题视为不可比(ok=None)")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    src = Path(args.report)
    report = json.loads(src.read_text(encoding="utf-8"))
    if "records" not in report:
        raise SystemExit(f"{src} 里没有 records —— 不是 run_eval.py 产出的报告")

    fields = {k for r in report["records"] for k in r}
    if args.field not in fields:
        raise SystemExit(
            f"字段 {args.field!r} 在这份报告里不存在。可用的逐题字段:\n  "
            + ", ".join(sorted(fields)))
    if args.gate and args.gate not in fields:
        raise SystemExit(f"门禁字段 {args.gate!r} 不存在")

    view, comparable = make_view(report, args.field, args.gate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")

    n_true = sum(1 for r in view["records"] if r["ok"] is True)
    print(f"{src.name}  --{args.field}-->  {out}")
    print(f"  可比题 {comparable} 道,其中 {args.field} 为真 {n_true} 道"
          f"  ({n_true / comparable:.1%})" if comparable else "  无可比题")
    if args.gate:
        dropped = sum(1 for r in view["records"] if r["ok"] is None)
        print(f"  被门禁 {args.gate} 排除 {dropped} 道")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
