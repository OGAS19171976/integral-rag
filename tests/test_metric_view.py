"""eval/metric_view.py 的测试。

重点测 `--gate` 这个设计,因为省掉它得到的是一个**看起来正常但结论错误**的
报告:没有期望卡片标注的题会被算进分母,比率被稀释,配对检验把两版都记为错。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import metric_view   # noqa: E402


def report(records):
    return {"summary": {}, "seconds_total": 1.0, "records": records}


SAMPLE = report([
    {"id": "d1", "expect_card": "c1", "card_hit_top1": True, "card_hit_top3": True},
    {"id": "d2", "expect_card": "c2", "card_hit_top1": False, "card_hit_top3": True},
    # 没有期望卡片标注 —— 检索指标在这道题上没有意义
    {"id": "d3", "expect_card": None, "card_hit_top1": False, "card_hit_top3": False},
])


def test_gate_excludes_items_without_expectation():
    view, comparable = metric_view.make_view(SAMPLE, "card_hit_top1", "expect_card")
    assert comparable == 2
    assert [r["ok"] for r in view["records"]] == [True, False, None]


def test_without_gate_the_denominator_is_wrong():
    """这条测试就是"为什么不能省掉 gate"的证据。

    不加门禁:3 道题里 1 道为真 → 33.3%;加了门禁:2 道题里 1 道 → 50.0%。
    差 16.7 个百分点,而报告本身不会有任何异常提示。
    """
    ungated, n_ungated = metric_view.make_view(SAMPLE, "card_hit_top1")
    gated, n_gated = metric_view.make_view(SAMPLE, "card_hit_top1", "expect_card")
    assert n_ungated == 3 and n_gated == 2
    assert sum(1 for r in ungated["records"] if r["ok"] is True) / n_ungated == pytest.approx(1 / 3)
    assert sum(1 for r in gated["records"] if r["ok"] is True) / n_gated == pytest.approx(1 / 2)


def test_none_stays_none_and_is_not_counted():
    src = report([{"id": "a", "value_match": None}, {"id": "b", "value_match": True}])
    view, comparable = metric_view.make_view(src, "value_match")
    assert comparable == 1
    assert [r["ok"] for r in view["records"]] == [None, True]


def test_make_view_does_not_mutate_the_input():
    before = json.dumps(SAMPLE, sort_keys=True)
    metric_view.make_view(SAMPLE, "card_hit_top3", "expect_card")
    assert json.dumps(SAMPLE, sort_keys=True) == before


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_cli_writes_a_usable_report(workdir, capsys):
    src = _write(workdir / "rep.json", SAMPLE)
    dst = workdir / "out" / "view.json"
    sys.argv = ["metric_view.py", str(src), "--field", "card_hit_top3",
                "--gate", "expect_card", "-o", str(dst)]
    assert metric_view.main() == 0
    out = capsys.readouterr().out
    assert "可比题 2 道" in out
    written = json.loads(dst.read_text(encoding="utf-8"))
    assert [r["ok"] for r in written["records"]] == [True, True, None]


def test_cli_rejects_a_field_that_does_not_exist(workdir):
    src = _write(workdir / "rep.json", SAMPLE)
    sys.argv = ["metric_view.py", str(src), "--field", "nope", "-o",
                str(workdir / "x.json")]
    with pytest.raises(SystemExit) as excinfo:
        metric_view.main()
    assert "nope" in str(excinfo.value)


def test_cli_rejects_a_report_without_records(workdir):
    src = _write(workdir / "rep.json", {"summary": {}})
    sys.argv = ["metric_view.py", str(src), "--field", "ok", "-o",
                str(workdir / "x.json")]
    with pytest.raises(SystemExit):
        metric_view.main()


def test_end_to_end_view_then_compare(workdir, capsys):
    """整条链路:报告 → 视图 → 配对比较,确认能得出正确结论。"""
    recs = []
    for i in range(40):
        recs.append({"id": f"q{i}", "expect_card": "c",
                     "card_hit_top1": i >= 12,        # 28 道命中
                     "card_hit_top3": True})          # 全部命中
    src = _write(workdir / "rep.json", report(recs))

    sys.argv = ["metric_view.py", str(src), "--field", "card_hit_top1",
                "--gate", "expect_card", "-o", str(workdir / "a.json")]
    metric_view.main()
    sys.argv = ["metric_view.py", str(src), "--field", "card_hit_top3",
                "--gate", "expect_card", "-o", str(workdir / "b.json")]
    metric_view.main()

    import compare
    a = json.loads((workdir / "a.json").read_text(encoding="utf-8"))
    b = json.loads((workdir / "b.json").read_text(encoding="utf-8"))
    res = compare.analyze_pair({r["id"]: r for r in a["records"]},
                               {r["id"]: r for r in b["records"]},
                               label_a="top-1", label_b="top-3", metrics=[("ok", "命中")],
                               n_boot=200)
    row = res.rows[0]
    assert row.n == 40
    assert row.mcnemar.a_only == 12      # 改前错、改后对
    assert row.mcnemar.b_only == 0
    assert row.verdict == compare.VERDICT_IMPROVED
