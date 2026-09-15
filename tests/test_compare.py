"""eval/compare.py 的测试。

比较器的输出会直接影响"要不要继续调、评估集还要不要扩"这种决定,
所以这里不能只测"能跑通",必须测**结论正确**:

  * 真正的改善 → 显著改善;
  * 两版逐题相同 → "逐题相同",**不是**"证据不足"(两者含义完全不同:
    前者是没有差异,后者是没测出来);
  * 只在一边存在的题、以及某一边为 None 的题 → 必须被排除,
    不能当成"另一边失败"。

分析逻辑是纯函数(analyze_pair),所以绝大多数用例不需要碰文件系统 ——
这既快,也躲开了受限环境里临时目录不可用的问题。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# eval/ 直接进 sys.path,这样 compare.py 是个正常的顶层模块。
# (不能用 importlib.util.module_from_spec 那种"不注册进 sys.modules"的加载法:
#  compare.py 里用了 @dataclass,而 dataclasses 要靠 sys.modules[cls.__module__]
#  去解析注解,模块不在表里就会抛 AttributeError。)
sys.path.insert(0, str(ROOT / "eval"))

import compare   # noqa: E402

S = compare.S


def recs(flags: dict, **extra) -> dict:
    out = {}
    for rid, val in flags.items():
        rec = {"id": rid, "ok": val}
        rec.update(extra)
        out[rid] = rec
    return out


def analyze(a: dict, b: dict, **kw):
    kw.setdefault("n_boot", 300)
    return compare.analyze_pair(a, b, label_a="A", label_b="B", **kw)


# ======================================================================
# 对齐:哪些题才算"可比"
# ======================================================================
def test_paired_flags_drops_items_missing_from_either_side():
    a = {"q1": {"ok": True}, "q2": {"ok": False}, "q3": {"ok": True}}
    b = {"q1": {"ok": False}, "q2": {"ok": True}, "q4": {"ok": True}}
    ids, fa, fb = compare.paired_flags(a, b, "ok")
    assert ids == ["q1", "q2"]
    assert fa == [True, False] and fb == [False, True]


def test_paired_flags_drops_none_on_either_side():
    """value_match 在"该题本就该发散"时是 None。

    那不是"某一版失败",而是这道题在这个指标上没有可比性 —— 整对丢掉。
    算进去会把两版都记为错,凭空制造差异。
    """
    a = {"d1": {"value_match": True}, "d2": {"value_match": None}, "d3": {"value_match": False}}
    b = {"d1": {"value_match": True}, "d2": {"value_match": False}, "d3": {"value_match": None}}
    ids, fa, fb = compare.paired_flags(a, b, "value_match")
    assert ids == ["d1"]
    assert len(fa) == len(fb) == 1


def test_pick_metrics_skips_fields_absent_from_either_report():
    a = {"q1": {"ok": True, "family_hit": True}}
    b = {"q1": {"ok": True}}
    assert [k for k, _ in compare.pick_metrics(a, b, None)] == ["ok"]


def test_pick_metrics_skips_fields_that_are_none_everywhere():
    a = {"q1": {"ok": True, "value_match": None}}
    b = {"q1": {"ok": True, "value_match": None}}
    assert [k for k, _ in compare.pick_metrics(a, b, None)] == ["ok"]


def test_metric_override():
    assert compare.parse_metric_override("ok,card_hit_top1") == [
        ("ok", "求解 / 给出答案"), ("card_hit_top1", "top-1 卡片命中")]
    assert compare.parse_metric_override("custom") == [("custom", "custom")]
    assert compare.parse_metric_override(None) is None


# ======================================================================
# 结论正确性
# ======================================================================
def test_real_improvement_is_significant():
    """A 全错 → B 全对(40 题):40 道净改善,必须显著改善。"""
    n = 40
    res = analyze(recs({f"q{i}": False for i in range(n)}),
                  recs({f"q{i}": True for i in range(n)}))
    row = res.rows[0]
    assert row.verdict == compare.VERDICT_IMPROVED
    assert row.mcnemar.delta == pytest.approx(1.0)
    assert row.mcnemar.discordant == n
    assert row.mcnemar.a_only == n and row.mcnemar.b_only == 0
    assert row.holm_p < 0.05
    assert row.n == n


def test_full_regression_is_flagged_as_degradation():
    n = 40
    res = analyze(recs({f"q{i}": True for i in range(n)}),
                  recs({f"q{i}": False for i in range(n)}))
    row = res.rows[0]
    assert row.verdict == compare.VERDICT_DEGRADED
    assert row.mcnemar.delta == pytest.approx(-1.0)
    assert row.mcnemar.b_only == n and row.mcnemar.a_only == 0


def test_identical_reports_say_identical_not_inconclusive():
    """"逐题完全相同"和"证据不足"是两回事。

    前者是差异为 0,样本再多也没用;后者是没测出来,扩样本可能有救。
    把它们印成同一句话,读者会去扩样本 —— 白花标注预算。
    """
    flags = {f"q{i}": i % 3 == 0 for i in range(30)}
    res = analyze(recs(flags), recs(flags))
    row = res.rows[0]
    assert row.verdict == compare.VERDICT_IDENTICAL
    assert row.mcnemar.discordant == 0
    assert row.mcnemar.exact_p == 1.0
    assert row.required_n == -1


def test_small_difference_is_inconclusive_and_quantifies_why():
    """56 题里净改善 3 道 —— 按检出能力门槛(6 道)必然不显著。

    关键不只是"判成不显著",还要能说出**为什么**、以及需要多少样本。
    """
    n = 56
    res = analyze(recs({f"q{i}": False for i in range(n)}),
                  recs({f"q{i}": i < 3 for i in range(n)}))
    row = res.rows[0]
    assert row.verdict == compare.VERDICT_INCONCLUSIVE
    assert row.mcnemar.a_only == 3 and row.mcnemar.b_only == 0
    assert row.holm_p > 0.05
    assert res.min_detectable_k == 6
    assert res.min_detectable_frac == pytest.approx(6 / 56)


def test_two_sided_churn_cancels_to_no_difference():
    """改好 10 道、同时改坏 10 道:比率完全没变,但系统其实变了。

    这正是只看"比率没变"会漏掉的东西 —— 分歧对是 20。
    """
    n = 40
    a = {f"q{i}": (i >= 20) for i in range(n)}      # 前 20 错、后 20 对
    b = {f"q{i}": (i < 20) for i in range(n)}       # 完全反过来
    res = analyze(recs(a), recs(b))
    row = res.rows[0]
    assert row.mcnemar.discordant == n
    assert row.mcnemar.delta == pytest.approx(0.0)
    assert row.verdict == compare.VERDICT_INCONCLUSIVE


def test_multiple_metrics_are_corrected_together():
    """多个指标各自"边缘显著"时,整体校正必须把它们压下去。

    这是本脚本默认做多重比较校正的意义:不校正的话,
    报 6 个指标时"至少一个假阳性"的概率接近 26%。
    """
    n = 60
    a, b = {}, {}
    for i in range(n):
        a[f"q{i}"] = {"id": f"q{i}", "ok": False, "family_hit": False,
                      "card_hit_top1": False}
        # 每个指标各改好 3 道,但改好的是**不同的**题
        b[f"q{i}"] = {"id": f"q{i}",
                      "ok": i < 3,
                      "family_hit": 3 <= i < 6,
                      "card_hit_top1": 6 <= i < 9}
    res = analyze(a, b)
    assert len(res.rows) == 3
    # 逐指标裸看 p 都不算太离谱,但校正后一个都不该显著
    assert all(r.verdict != compare.VERDICT_IMPROVED for r in res.rows)
    assert all(r.holm_p >= r.mcnemar.exact_p for r in res.rows)


def test_homogeneous_metrics_stay_significant_after_correction():
    """对照组:改善幅度足够大时,校正不该把真效果也一起压掉。

    (如果上个测试只用"一个都不显著"来断言,那么一个永远返回 p=1 的
     实现也能通过 —— 这个用例就是为了挡住那种情况。)
    """
    n = 60
    a, b = {}, {}
    for i in range(n):
        a[f"q{i}"] = {"id": f"q{i}", "ok": False, "family_hit": False,
                      "card_hit_top1": False}
        good = i < 30
        b[f"q{i}"] = {"id": f"q{i}", "ok": good, "family_hit": good,
                      "card_hit_top1": good}
    res = analyze(a, b)
    assert len(res.rows) == 3
    assert all(r.verdict == compare.VERDICT_IMPROVED for r in res.rows)


def test_disjoint_ids_are_excluded_and_listed():
    res = analyze(recs({f"a{i}": True for i in range(10)}),
                  recs({f"b{i}": True for i in range(10)}))
    assert not res.comparable
    assert res.only_a and res.only_b
    assert "没有任何两边都可比的指标" in compare.render_text(res)


def test_partial_overlap_compares_only_shared():
    shared = {f"s{i}": True for i in range(20)}
    res = analyze(recs({**shared, "x1": True, "x2": True}),
                  recs({**shared, "y1": False}))
    assert res.rows[0].n == 20
    assert res.rows[0].verdict == compare.VERDICT_IDENTICAL
    assert res.only_a == ["x1", "x2"] and res.only_b == ["y1"]


def test_sample_size_hint_is_reported_for_inconclusive_rows():
    """不显著时,必须给出"要多少样本"这个可执行的数字。"""
    n = 40
    a = recs({f"q{i}": False for i in range(n)})
    b = recs({f"q{i}": i < 4 for i in range(n)})
    res = analyze(a, b, n_boot=200)
    row = res.rows[0]
    assert row.verdict == compare.VERDICT_INCONCLUSIVE
    assert row.required_n > n, "要检出必须得比现在多"
    text = compare.render_text(res)
    assert "约需" in text


# ======================================================================
# 渲染
# ======================================================================
def test_text_report_contains_the_key_sections():
    n = 40
    res = analyze(recs({f"q{i}": False for i in range(n)}),
                  recs({f"q{i}": True for i in range(n)}), n_boot=200)
    text = compare.render_text(res, "a.json", "b.json", 1.0, 2.0)
    assert "表 1" in text and "表 2" in text
    assert "检出能力" in text
    assert "95% Wilson" in text
    assert "净改善至少要 6 道题" in text
    assert "Holm p" in text


def test_markdown_report_is_well_formed():
    n = 40
    res = analyze(recs({f"q{i}": False for i in range(n)}),
                  recs({f"q{i}": True for i in range(n)}), n_boot=200)
    md = compare.render_markdown(res, "a.json", "b.json")
    assert md.startswith("# 配对比较")
    assert "| 指标 |" in md
    assert "显著改善" in md
    assert "检出能力" in md
    # 表头与分隔行的列数必须一致,否则 markdown 表格会渲染错乱
    header = [ln for ln in md.splitlines() if ln.startswith("| 指标 |")][0]
    sep = [ln for ln in md.splitlines() if ln.startswith("|---")][0]
    assert header.count("|") == sep.count("|")


# ======================================================================
# 文件入口(CLI)
# ======================================================================
def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _report(flags: dict, seconds: float = 1.0) -> dict:
    return {"summary": {}, "seconds_total": seconds,
            "records": [{"id": k, "ok": v} for k, v in flags.items()]}


def test_cli_end_to_end(workdir, monkeypatch, capsys):
    n = 40
    pa = _write(workdir / "a.json", _report({f"q{i}": False for i in range(n)}))
    pb = _write(workdir / "b.json", _report({f"q{i}": True for i in range(n)}))
    md = workdir / "out" / "report.md"
    monkeypatch.setattr(sys, "argv", [
        "compare.py", str(pa), str(pb), "--label-a", "k=4", "--label-b", "k=8",
        "--n-boot", "200", "--md", str(md)])
    assert compare.main() == 0
    out = capsys.readouterr().out
    assert "k=4" in out and "k=8" in out
    assert compare.VERDICT_IMPROVED in out
    assert md.exists()
    assert "# 配对比较" in md.read_text(encoding="utf-8")


def test_cli_returns_1_when_nothing_comparable(workdir, monkeypatch, capsys):
    pa = _write(workdir / "a.json", _report({f"a{i}": True for i in range(5)}))
    pb = _write(workdir / "b.json", _report({f"b{i}": True for i in range(5)}))
    monkeypatch.setattr(sys, "argv", ["compare.py", str(pa), str(pb)])
    assert compare.main() == 1
    assert "没有任何两边都可比的指标" in capsys.readouterr().out


def test_cli_rejects_json_without_records(workdir, monkeypatch, capsys):
    bad = _write(workdir / "bad.json", {"summary": {}})
    good = _write(workdir / "good.json", _report({"q1": True}))
    monkeypatch.setattr(sys, "argv", ["compare.py", str(bad), str(good)])
    with pytest.raises(SystemExit):
        compare.main()


# ======================================================================
# 检出能力门槛(在 stats 里实现,这里验证它的"人话含义")
# ======================================================================
def test_minimum_detectable_delta_hand_computed():
    """手算:净改善 k 道、反向 0 道时精确 p = 2·(1/2)^k。
       k=5 → 0.0625(不显著);k=6 → 0.03125(显著)。门槛 = 6。
    """
    k, frac = S.minimum_detectable_delta(56)
    assert k == 6
    assert frac == pytest.approx(6 / 56)
    assert 2 * 0.5 ** 5 >= 0.05 > 2 * 0.5 ** 6


def test_detectable_count_does_not_depend_on_total_size():
    """评估集从 50 题扩到 1000 题,能检出的**净改善题数门槛**一点没变。

    变的只是它占的百分比。这是"单纯扩大评估集不提高判别力"的量化版本 ——
    也是标注预算最容易被浪费掉的地方。
    """
    k_small, frac_small = S.minimum_detectable_delta(50)
    k_big, frac_big = S.minimum_detectable_delta(1000)
    assert k_small == k_big == 6
    assert frac_big < frac_small / 15


def test_detectable_delta_returns_minus_one_when_impossible():
    k, frac = S.minimum_detectable_delta(3)
    assert k == -1 and math.isnan(frac)
    k0, frac0 = S.minimum_detectable_delta(0)
    assert k0 == -1 and math.isnan(frac0)
