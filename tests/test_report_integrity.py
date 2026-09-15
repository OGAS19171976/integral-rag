"""评估报告的内部一致性。

背景(实测撞到,不是假想):`run_eval.py` 曾经产出一份这样的报告 ——

    summary :  "value_match": 51
    records :  56 道题里逐题 value_match 为 True 的 **0 道**

两个数字直接互相矛盾,而报告看起来完全正常。下游任何按题分析
(配对检验、消融实验、失败归因)读的都是**逐题字段**,于是结论全错,
而且没有任何地方会报错。

根因是在 `record.update({...})` 的字典字面量里写了
`record.get("ok", False)` —— 读的是**这次 update 之前**的字典,永远是默认值。

所以这里锁两件事:

  1. `_audit_summary` 真的能抓住这类不一致(而不是永远放行);
  2. 仓库里提交的报告文件**现在**是一致的。

第 2 条会随着重新生成报告而变,这正是它的价值:以后谁改坏了生成逻辑,
提交一份自相矛盾的报告,这个测试会立刻拦下来。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval   # noqa: E402


# ======================================================================
# 自检本身有效吗
# ======================================================================
def test_audit_passes_on_consistent_data():
    stats = {"solved": 2, "family_hit": 1, "card_hit_top1": 1, "card_hit_top3": 2}
    records = [
        {"id": "a", "ok": True, "family_hit": True, "card_hit_top1": True,
         "card_hit_top3": True},
        {"id": "b", "ok": True, "family_hit": False, "card_hit_top1": False,
         "card_hit_top3": True},
    ]
    run_eval._audit_summary(stats, records)      # 不应抛异常


def test_audit_catches_the_exact_bug_we_hit():
    """复刻真实故障:summary 说 51/51,逐题字段全是 False。"""
    stats = {"total": 56, "solved": 51, "value_match": 51}
    records = [{"id": f"d{i:02d}", "kind": "definite", "expect": "1/3",
                "ok": True, "value_match": False} for i in range(51)]
    with pytest.raises(SystemExit) as excinfo:
        run_eval._audit_summary(stats, records)
    message = str(excinfo.value)
    assert "value_match" in message
    assert "51" in message and "0" in message


def test_audit_catches_false_finite():
    """把发散报成有限值是这套系统最危险的错误,它的计数也必须对得上账。"""
    stats = {"expect_divergent": 2, "divergence_ok": 1}
    records = [
        {"id": "x", "kind": "definite", "expect": "divergent", "diverges": True},
        {"id": "y", "kind": "definite", "expect": "divergent", "diverges": True},
    ]
    with pytest.raises(SystemExit) as excinfo:
        run_eval._audit_summary(stats, records)
    assert "divergence_ok" in str(excinfo.value)


def test_audit_ignores_fields_missing_from_summary():
    """定积分报告里没有 family_hit 之类的键时不该报错,更不该凭空造一个检查。"""
    run_eval._audit_summary({"solved": 1}, [{"id": "a", "ok": True}])


def test_audit_excludes_divergent_items_from_solved():
    """定积分里 expect == "divergent" 的题不计入 solved。

    判据写错会让自检本身误报 —— 一个总在报警的自检等于没有自检。
    """
    stats = {"solved": 1}
    records = [
        {"id": "ok1", "kind": "definite", "expect": "1/3", "ok": True},
        {"id": "div", "kind": "definite", "expect": "divergent", "ok": True},
    ]
    run_eval._audit_summary(stats, records)

    with pytest.raises(SystemExit):
        run_eval._audit_summary({"solved": 2}, records)


def test_audit_counts_failed_records_as_not_solved():
    """解析失败/引擎异常也会进 records,它们不该被算成 solved。"""
    stats = {"solved": 1}
    records = [
        {"id": "a", "ok": True},
        {"id": "b", "ok": False, "stage": "parse", "error": "boom"},
    ]
    run_eval._audit_summary(stats, records)


def test_audit_is_called_by_dump(workdir):
    """自检必须真的挂在出报告这条路上,否则它只是一段没人调用的代码。"""
    bad_stats = {"solved": 5}
    bad_records = [{"id": "a", "ok": False}]

    class Args:
        json_out = str(workdir / "out.json")

    with pytest.raises(SystemExit):
        run_eval._dump(Args(), bad_stats, 1.0, bad_records)
    assert not (workdir / "out.json").exists(), "自检失败时不该写出报告"


def test_dump_writes_when_consistent(workdir):
    """对照组:数据一致时必须正常出报告(挡住"永远报错"的实现)。"""
    class Args:
        json_out = str(workdir / "sub" / "ok.json")

    run_eval._dump(Args(), {"solved": 1}, 1.0, [{"id": "a", "ok": True}])
    written = json.loads((workdir / "sub" / "ok.json").read_text(encoding="utf-8"))
    assert written["summary"] == {"solved": 1}
    assert len(written["records"]) == 1


# ======================================================================
# 仓库里提交的报告是否一致
# ======================================================================
@pytest.mark.parametrize("name", ["report.json", "report_definite.json"])
def test_committed_reports_are_internally_consistent(name):
    path = ROOT / "eval" / name
    if not path.exists():
        pytest.skip(f"{name} 不存在")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 不一致会抛 SystemExit,这里就是断言
    run_eval._audit_summary(data["summary"], data["records"])


@pytest.mark.parametrize("name", ["report.json", "report_definite.json"])
def test_committed_reports_have_usable_per_item_fields(name):
    """逐题字段必须"有区分度"。

    一个恒为 False 的字段(正是我们踩过的那个 bug)在数值上可能凑巧和
    summary 对得上,但在下游完全无用。所以这里额外要求:凡是 summary 里
    计了数的指标,逐题字段必须有正例也有负例(除非真的全对/全错)。
    """
    path = ROOT / "eval" / name
    if not path.exists():
        pytest.skip(f"{name} 不存在")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    for key, total in data["summary"].items():
        if key in ("total",) or not isinstance(total, int) or total <= 0:
            continue
        values = [r.get(key) for r in records if key in r]
        if not values:
            continue
        trues = sum(1 for v in values if v is True)
        assert trues == total, f"{name}:{key} 逐题 True 数 {trues} != summary {total}"
