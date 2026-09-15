"""评估脚本:在一套带标注的题目上量出这套系统的真实水平。

不定积分与定积分分开测,因为它们的失败含义不同:

  不定积分
    1. 求解率 —— 给出并通过**求导回验**的比例。
    2. 检索命中率 —— top-1 卡片的方法族是否落在预期集合里。
    3. 拒答数 —— 未能给出通过验证的结果(不是"错",是系统选择不作答)。
    4. 结果可读性 —— ugliness 高于阈值的比例。

  定积分
    1. 求解率 —— 给出并通过**高精度数值求积**验证的比例。
    2. 与预期闭式一致率 —— 拿题目里标注的**独立期望值**再比一次。
       这一步很关键:数值层同时充当了"验证器"和"真值来源",
       如果只用它验证,就存在自证的风险。用标注值做第三方比对才是真的交叉检验。
    3. 发散判定正确率 —— 该发散的题有没有正确判成发散
       (这是定积分最危险的一类错误:把发散当成一个有限值报出去)。
    4. 检索命中率 / 耗时。

用法:
    python eval/run_eval.py                    # 不定积分
    python eval/run_eval.py --definite         # 定积分与含参积分
    python eval/run_eval.py --verbose          # 打印失败题详情
    python eval/run_eval.py --json out.json    # 结果写到文件
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integral_rag import analyze                                  # noqa: E402
from integral_rag.parse import SYMBOLS as PARSE_SYMBOLS            # noqa: E402
from integral_rag.parse import ParseError                          # noqa: E402
from integral_rag.quadrature import close                          # noqa: E402
from integral_rag.retrieve import Retriever                        # noqa: E402
from integral_rag.solve import CLEAN_THRESHOLD, ugliness           # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR / "testset.jsonl"
TESTSET_DEFINITE = EVAL_DIR / "testset_definite.jsonl"
CORPUS = ROOT / "corpus" / "method_cards.json"
CORPUS_DEFINITE = ROOT / "corpus" / "definite_cards.json"

# 期望值里的符号**必须复用解析器那一份**。
# 自己再声明一遍很容易漏掉某个字母(原来漏了 s),于是 gamma(s) 里的 s
# 变成一个没有假设的新符号,和结果里的 s 不是同一个对象,
# 代不进去 → 比对报"数值化失败",明明是相等的两个式子被判成不符。
SYMBOLS = dict(PARSE_SYMBOLS)


def load_testset(path: Path) -> list[dict]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no} 不是合法 JSON:{exc}") from exc
    return items


def family_of(card_id: str | None, cards_by_id: dict[str, dict]) -> str | None:
    if not card_id:
        return None
    card = cards_by_id.get(card_id)
    return card.get("family") if card else None


# ================================================================ 不定积分
def run_indefinite(items: list[dict], retriever, args) -> int:
    cards_by_id = {c.get("id"): c for c in retriever.cards}

    print("=" * 104)
    print(f"评估集 {len(items)} 题(不定积分)    方法卡片语料 {len(retriever.cards)} 张")
    print("=" * 104)
    print(f"{'id':<6}{'题目':<34}{'求解':<6}{'top-1 卡片':<32}{'族命中':<8}{'尝试':<6}{'秒':<6}")
    print("-" * 104)

    records, stats, fam_counter = [], Counter(), Counter()
    miss_family, failures = [], []
    total_time = 0.0

    for item in items:
        latex = item["latex"]
        expect_families = item.get("family") or []
        expect_card = item.get("card")
        started = time.time()
        record = {"id": item["id"], "latex": latex, "kind": "indefinite",
                  "expect_family": expect_families, "expect_card": expect_card}
        try:
            report = analyze(latex, retriever=retriever, k=args.k)
        except ParseError as exc:
            elapsed = time.time() - started
            total_time += elapsed
            stats["parse_fail"] += 1
            print(f"{item['id']:<6}{latex[:32]:<34}{'解析失败':<6}")
            record.update({"ok": False, "stage": "parse", "error": str(exc)})
            records.append(record)
            failures.append(record)
            continue
        except Exception as exc:
            elapsed = time.time() - started
            total_time += elapsed
            stats["engine_error"] += 1
            print(f"{item['id']:<6}{latex[:32]:<34}{'引擎异常':<6}")
            record.update({"ok": False, "stage": "engine",
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
            records.append(record)
            failures.append(record)
            continue

        elapsed = time.time() - started
        total_time += elapsed
        top1 = report.top1_card_id
        top1_family = family_of(top1, cards_by_id)
        top3_ids = [h.id for h in report.hits[:3]]
        clean = report.ok and ugliness(report.result) <= CLEAN_THRESHOLD
        fam_hit = bool(expect_families) and top1_family in expect_families
        card_hit_top1 = bool(expect_card) and top1 == expect_card
        card_hit_top3 = bool(expect_card) and expect_card in top3_ids

        stats["total"] += 1
        stats["solved"] += report.ok
        stats["refused"] += (not report.ok)
        stats["family_hit"] += fam_hit
        stats["family_total"] += bool(expect_families)
        stats["card_hit_top1"] += card_hit_top1
        stats["card_hit_top3"] += card_hit_top3
        stats["card_total"] += bool(expect_card)
        stats["clean"] += clean
        stats["piecewise"] += bool(report.ok and report.result is not None
                                   and report.result.has(sp.Piecewise))
        fam_counter[top1_family] += 1
        if not fam_hit and expect_families:
            miss_family.append(f"{item['id']} {latex} → {top1}({top1_family}) 期望 {expect_families}")

        record.update({
            "ok": bool(report.ok),
            "result": str(report.result) if report.ok else None,
            "strategy": report.solution.strategy,
            "proof": report.solution.proof,
            "top1_card": top1, "top1_family": top1_family,
            "top_hits": [h.id for h in report.hits],
            "family_hit": fam_hit, "card_hit_top1": card_hit_top1,
            "card_hit_top3": card_hit_top3,
            "n_attempts": len(report.solution.attempts),
            "ugliness": round(ugliness(report.result), 2) if report.ok else None,
            "seconds": round(elapsed, 2),
        })
        records.append(record)
        if not report.ok:
            failures.append(record)

        show = latex if len(latex) <= 32 else latex[:29] + "..."
        print(f"{item['id']:<6}{show:<34}{('✅' if report.ok else '❌'):<6}"
              f"{(top1 or '-'):<32}"
              f"{('✓' if fam_hit else ('✗' if expect_families else '-')):<8}"
              f"{len(report.solution.attempts):<6}{elapsed:<6.2f}")

    n = max(stats["total"], 1)
    print("=" * 104)
    print("汇总")
    print("-" * 104)
    print(f"  题目总数            {stats['total']}")
    print(f"  解析失败 / 引擎异常  {stats['parse_fail']} / {stats['engine_error']}")
    print(f"  求解并通过验证      {stats['solved']}/{stats['total']}   ({stats['solved'] / n:.1%})")
    print(f"  拒答(未给出答案)    {stats['refused']}/{stats['total']}"
          f"   ({stats['refused'] / n:.1%})   ← 设计上的诚实,不是错误")
    if stats["family_total"]:
        print(f"  top-1 方法族命中    {stats['family_hit']}/{stats['family_total']}"
              f"   ({stats['family_hit'] / max(stats['family_total'], 1):.1%})")
    if stats["card_total"]:
        print(f"  top-1 卡片命中      {stats['card_hit_top1']}/{stats['card_total']}"
              f"   ({stats['card_hit_top1'] / max(stats['card_total'], 1):.1%})")
        print(f"  top-3 卡片命中      {stats['card_hit_top3']}/{stats['card_total']}"
              f"   ({stats['card_hit_top3'] / max(stats['card_total'], 1):.1%})")
    if stats["solved"]:
        print(f"  结果形式干净        {stats['clean']}/{stats['solved']}"
              f"   ({stats['clean'] / max(stats['solved'], 1):.1%})   ← ugliness ≤ {CLEAN_THRESHOLD}")
    print(f"  总耗时              {total_time:.1f} 秒(平均 {total_time / n:.2f} 秒/题)")

    if miss_family:
        print(f"\n方法族未命中的题目({len(miss_family)} 道):")
        for line in miss_family:
            print(f"    {line}")
    if failures and args.verbose:
        print(f"\n未给出答案的题目({len(failures)} 道):")
        for record in failures:
            print(f"\n  [{record['id']}] {record['latex']}")
            print(f"      检索 top-{args.k}:{record.get('top_hits')}")
            if record.get("error"):
                print(f"      错误:{record['error']}")
    print(f"\n最常被推荐的方法族:{dict(fam_counter.most_common(8))}")
    _dump(args, stats, total_time, records)
    return 0


# ================================================================ 定积分
def _parse_expect(text: str) -> sp.Expr | None:
    try:
        return sp.sympify(text, locals=dict(SYMBOLS))
    except Exception:
        return None


def _check_value(value, expect, f, x, forced_params=None):
    """把符号结果和标注的期望闭式在若干组参数取值上比对。

    这是**独立于数值层**的第三方检查:数值层既当验证器又当真值来源,
    只靠它验证有自证的风险。

    参数集合必须取"被积函数 ∪ 期望闭式"两边的自由符号。只从被积函数取会漏掉
    参数只出现在**积分限**里的情形(∫_{-a}^{a} x²dx 的参数 a 就只在限里),
    那样参数代不进去,比对会误报"不符" —— 实测就是这样误报了 3 道题。
    """
    from integral_rag.definite import _parameter_combos

    if value is None:
        return False, float("inf"), "没有结果"
    params = sorted({s for s in (value.free_symbols | expect.free_symbols)
                     if s is not x and s.is_real}, key=lambda s: s.name)
    if forced_params:
        combos = [forced_params]
    elif params:
        combos = _parameter_combos(params)
    else:
        combos = [{}]

    worst = 0.0
    # 参数默认取正 —— 这既是教科书约定,也是本系统 parse 的假设
    # (parse.SYMBOLS 把参数标成 positive)。
    # 只比正数采样点,否则闭式会因为在 t<0 上的主值分支符号差异被误报成
    # "不符"。实测 ∫_0^∞ e^{-t x²}dx 就是这样:系统给的
    # sqrt(pi)/(2 sqrt(t)) 与标注的 sqrt(pi/t)/2 在 t>0 上恒等,
    # 但 t=-0.5 时一个是 -1.2533i、另一个是 +1.2533i,
    # 于是这道本来完全正确的题被计成 value_match=False。
    # 仍然是**所有**正数采样点都必须一致,所以不会放过真正写错的闭式。
    positives = [c for c in combos if all(v > 0 for v in c.values())]
    sequence = positives or combos

    for combo in sequence:
        subs = {k: sp.Float(str(v), 30) for k, v in combo.items()}
        try:
            got = complex(sp.N(value.subs(subs), 30))
            want = complex(sp.N(expect.subs(subs), 30))
        except Exception as exc:
            return False, float("inf"), f"数值化失败:{type(exc).__name__}: {exc}"
        # 两边都发散(比如 π/sin(πs) 与它自己在 s=2 处都是 nan)→ 这个采样点没有信息量,
        # 跳过而不是判"不符"。否则会把一个正确的闭式报成不匹配。
        if not (math.isfinite(got.real) and math.isfinite(got.imag)) or \
           not (math.isfinite(want.real) and math.isfinite(want.imag)):
            continue
        ok, rel = close(got, want, rel_tol=1e-7)
        if not ok:
            return False, rel, f"{_fmt_combo(combo)} 期望 {want:.12g} 实得 {got:.12g}"
        worst = max(worst, rel)
    return True, worst, ""


def _fmt_combo(combo: dict) -> str:
    return "(" + ", ".join(f"{k}={v}" for k, v in combo.items()) + ")" if combo else "(无参数)"


def run_definite(items: list[dict], retriever, args) -> int:
    cards_by_id = {c.get("id"): c for c in retriever.cards}

    print("=" * 104)
    print(f"评估集 {len(items)} 题(定积分 / 含参积分)    方法卡片语料 {len(retriever.cards)} 张")
    print("=" * 104)
    print(f"{'id':<6}{'题目':<34}{'求解':<6}{'值一致':<8}{'top-1 卡片':<32}{'族命中':<8}{'秒':<6}")
    print("-" * 104)

    records, stats, fam_counter = [], Counter(), Counter()
    miss_family, failures, value_mismatch = [], [], []
    total_time = 0.0

    for item in items:
        latex = item["latex"]
        expect_families = item.get("family") or []
        expect_card = item.get("card")
        expect_raw = item.get("expect", "")
        expect_divergent = (expect_raw == "divergent")
        expect_expr = None if expect_divergent else _parse_expect(expect_raw)

        started = time.time()
        record = {"id": item["id"], "latex": latex, "kind": "definite",
                  "expect": expect_raw, "expect_family": expect_families,
                  "expect_card": expect_card, "note": item.get("note", "")}
        try:
            report = analyze(latex, retriever=retriever, k=args.k)
        except Exception as exc:
            elapsed = time.time() - started
            total_time += elapsed
            stats["engine_error"] += 1
            print(f"{item['id']:<6}{latex[:32]:<34}{'异常':<6}")
            record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
            records.append(record)
            failures.append(record)
            continue

        elapsed = time.time() - started
        total_time += elapsed
        top1 = report.top1_card_id
        top1_family = family_of(top1, cards_by_id)
        top3_ids = [h.id for h in report.hits[:3]]
        fam_hit = bool(expect_families) and top1_family in expect_families
        card_hit_top1 = bool(expect_card) and top1 == expect_card
        card_hit_top3 = bool(expect_card) and expect_card in top3_ids
        diverges = bool(getattr(report, "diverges", False))

        stats["total"] += 1
        stats["family_hit"] += fam_hit
        stats["family_total"] += bool(expect_families)
        stats["card_hit_top1"] += card_hit_top1
        stats["card_hit_top3"] += card_hit_top3
        stats["card_total"] += bool(expect_card)
        fam_counter[top1_family] += 1

        value_note = ""
        if expect_divergent:
            stats["expect_divergent"] += 1
            divergence_ok = diverges
            stats["divergence_ok"] += divergence_ok
            if not divergence_ok:
                stats["divergence_wrong"] += 1
                # 最危险的一类错误:该发散却报了一个有限值
                if report.ok:
                    stats["false_finite"] += 1
                value_note = "该发散却没判成发散"
                value_mismatch.append(f"{item['id']} {latex} → 结果 {report.result},应为发散")
        else:
            stats["expect_finite"] += 1
            if diverges:
                stats["wrongly_divergent"] += 1
                value_note = "误判为发散"
                value_mismatch.append(f"{item['id']} {latex} → 判为发散,实际收敛")
            elif report.ok:
                stats["solved"] += 1
                matched, rel, detail = _check_value(
                    report.result, expect_expr, report.f, report.x,
                    forced_params=item.get("params"))
                stats["value_match"] += matched
                if not matched:
                    value_note = f"值与期望不符({detail})"
                    value_mismatch.append(
                        f"{item['id']} {latex} → {report.result} vs 期望 {expect_raw}"
                        f"  [{detail}]")
            else:
                value_note = "未能给出结果"

        record.update({
            "ok": bool(report.ok),
            "diverges": diverges,
            "result": str(report.result) if report.ok else None,
            "strategy": getattr(report.solution, "strategy", ""),
            "proof": getattr(report.solution, "proof", ""),
            "conditions": list(getattr(report.solution, "conditions", [])),
            "top1_card": top1, "top1_family": top1_family,
            "top_hits": [h.id for h in report.hits],
            "family_hit": fam_hit, "card_hit_top1": card_hit_top1,
            "card_hit_top3": card_hit_top3,
            "value_match": None if expect_divergent else record.get("ok", False) and not value_note,
            "numeric": (report.solution.numeric.describe()
                        if getattr(report.solution, "numeric", None) else ""),
            "seconds": round(elapsed, 2),
        })
        records.append(record)
        if value_note:
            failures.append(record)

        show = latex if len(latex) <= 32 else latex[:29] + "..."
        solved_mark = ("✅" if report.ok else ("⛔" if diverges else "❌"))
        value_mark = "—" if expect_divergent else ("✓" if not value_note else "✗")
        if expect_divergent:
            value_mark = "✓" if diverges else "✗"
        print(f"{item['id']:<6}{show:<34}{solved_mark:<6}{value_mark:<8}"
              f"{(top1 or '-'):<32}"
              f"{('✓' if fam_hit else ('✗' if expect_families else '-')):<8}{elapsed:<6.2f}")
        if value_note and args.verbose:
            print(f"      → {value_note}")

    n = max(stats["total"], 1)
    print("=" * 104)
    print("汇总")
    print("-" * 104)
    print(f"  题目总数              {stats['total']}")
    print(f"  引擎异常              {stats['engine_error']}")
    if stats["expect_finite"]:
        print(f"  收敛题给出结果        {stats['solved']}/{stats['expect_finite']}"
              f"   ({stats['solved'] / max(stats['expect_finite'], 1):.1%})")
        print(f"  与标注闭式一致        {stats['value_match']}/{stats['expect_finite']}"
              f"   ({stats['value_match'] / max(stats['expect_finite'], 1):.1%})"
              f"   ← 独立于数值层的第三方比对")
        print(f"  误判为发散            {stats['wrongly_divergent']}")
    if stats["expect_divergent"]:
        print(f"  发散判定正确          {stats['divergence_ok']}/{stats['expect_divergent']}"
              f"   ({stats['divergence_ok'] / max(stats['expect_divergent'], 1):.1%})")
        print(f"  把发散报成有限值      {stats['false_finite']}"
              f"   ← 这一类错误最危险")
    if stats["family_total"]:
        print(f"  top-1 方法族命中      {stats['family_hit']}/{stats['family_total']}"
              f"   ({stats['family_hit'] / max(stats['family_total'], 1):.1%})")
    if stats["card_total"]:
        print(f"  top-1 卡片命中        {stats['card_hit_top1']}/{stats['card_total']}"
              f"   ({stats['card_hit_top1'] / max(stats['card_total'], 1):.1%})")
        print(f"  top-3 卡片命中        {stats['card_hit_top3']}/{stats['card_total']}"
              f"   ({stats['card_hit_top3'] / max(stats['card_total'], 1):.1%})")
    print(f"  总耗时                {total_time:.1f} 秒(平均 {total_time / n:.2f} 秒/题)")

    if miss_family:
        print(f"\n方法族未命中({len(miss_family)} 道):")
        for line in miss_family:
            print(f"    {line}")
    if value_mismatch:
        print(f"\n结果与期望不符({len(value_mismatch)} 道):")
        for line in value_mismatch:
            print(f"    {line}")
    if args.verbose:
        for record in failures:
            if record.get("error"):
                print(f"\n  [{record['id']}] 引擎异常:{record['error']}")
    print(f"\n最常被推荐的方法族:{dict(fam_counter.most_common(8))}")
    _dump(args, stats, total_time, records)
    return 0


def _dump(args, stats, total_time, records) -> None:
    if not args.json_out:
        return
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": dict(stats),
        "seconds_total": round(total_time, 2),
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已写入 {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="评估这套积分系统的实际水平")
    parser.add_argument("--definite", action="store_true",
                        help="评估定积分与含参积分")
    parser.add_argument("--testset", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题(0 表示全部)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args()

    path = Path(args.testset) if args.testset else (
        TESTSET_DEFINITE if args.definite else TESTSET)
    items = load_testset(path)
    if args.limit:
        items = items[: args.limit]

    if args.definite:
        retriever = Retriever.from_json(CORPUS_DEFINITE, definite=True)
        return run_definite(items, retriever, args)
    retriever = Retriever.from_json(CORPUS)
    return run_indefinite(items, retriever, args)


if __name__ == "__main__":
    raise SystemExit(main())
