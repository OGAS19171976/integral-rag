"""方法卡片语料的构建与自动校验(不定积分 + 定积分/含参积分通用)。

这个脚本回答三个问题:
  1. 语料格式对不对?(schema 校验 —— 字段、枚举值、priority 区间)
  2. 每张卡片自带的例题,是不是真的会被检索回它自己?(检索自命中率)
  3. 这些例题能不能真的算出来并通过验证?(语料可用性)

第 2、3 点是这套东西的关键:它把"语料写得好不好"变成了一个可以自动跑出来的
数字。卡片可以外包给人写、给模型写,但质量由这个脚本兜底 —— 写错的卡片
会立刻暴露在自命中率和求解率上。

定积分语料还多一件事:例题可能是**故意发散**的(divergence_test 族的卡
就用发散的积分当例子)。所以这里把"算出有限值"和"正确判出发散"都算作
"得到了明确结论",但分开统计。

用法:
    python build_corpus.py                 # 校验不定积分语料
    python build_corpus.py --definite      # 校验定积分/含参积分语料
    python build_corpus.py --quiet         # 只打印有问题的行
    python build_corpus.py --no-solve      # 跳过求解自测(快很多)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from integral_rag import analyze                       # noqa: E402
from integral_rag.retrieve import Retriever            # noqa: E402
from integral_rag.schema import validate_corpus        # noqa: E402

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus" / "method_cards.json"
DEFINITE_CORPUS = ROOT / "corpus" / "definite_cards.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="校验方法卡片语料")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--definite", action="store_true",
                        help="校验定积分/含参积分语料")
    parser.add_argument("--quiet", action="store_true", help="只显示有问题的卡片")
    parser.add_argument("--no-solve", action="store_true", help="跳过求解自测(快很多)")
    args = parser.parse_args()

    path = Path(args.corpus) if args.corpus else (
        DEFINITE_CORPUS if args.definite else CORPUS)
    if not path.exists():
        print(f"[错误] 找不到语料文件:{path}")
        return 2

    cards = json.loads(path.read_text(encoding="utf-8"))
    kind = "定积分/含参积分" if args.definite else "不定积分"
    print(f"语料({kind}):{path}")
    print(f"卡片数:{len(cards) if isinstance(cards, list) else '不是数组'}")
    print("=" * 78)

    # ---------------- 1. schema 校验 ----------------
    errors, warnings = validate_corpus(cards, definite=args.definite)
    if errors:
        print(f"\n❌ schema 校验发现 {len(errors)} 个错误:")
        for e in errors:
            print(f"   · {e}")
    else:
        print("\n✅ schema 校验通过")

    for w in warnings:
        print(f"\n⚠ {w}")

    if errors or not isinstance(cards, list):
        print("\n先修 schema 错误,再跑检索/求解自测。")
        return 1

    # ---------------- 2 & 3. 检索自命中 + 求解自测 ----------------
    retriever = Retriever(cards, definite=args.definite)
    family_of = {c.get("id"): c.get("family") for c in cards}

    print("\n" + "=" * 78)
    print(f"{'卡片 id':<36}{'解析':>8}{'top1':>8}{'top3':>8}{'结论':>8}")
    print("-" * 78)

    totals = {"examples": 0, "parsed": 0, "top1": 0, "top3": 0,
              "solved": 0, "diverged": 0, "undecided": 0}
    problems: list[str] = []
    same_family: list[str] = []
    cross_family: list[str] = []

    for card in cards:
        cid = card.get("id", "?")
        own_family = card.get("family")
        examples = card.get("examples") or []
        stat = {"examples": len(examples), "parsed": 0, "top1": 0, "top3": 0,
                "solved": 0, "diverged": 0, "undecided": 0}
        details: list[str] = []

        for raw in examples:
            totals["examples"] += 1
            try:
                report = analyze(raw, retriever=retriever, k=4)
            except Exception as exc:
                details.append(f"      ✗ {raw}  → 解析失败:{type(exc).__name__}: {exc}")
                continue

            stat["parsed"] += 1
            totals["parsed"] += 1

            top_ids = [h.id for h in report.hits]
            top1_family = family_of.get(top_ids[0]) if top_ids else None

            if top_ids and top_ids[0] == cid:
                stat["top1"] += 1
                totals["top1"] += 1
            if cid in top_ids[:3]:
                stat["top3"] += 1
                totals["top3"] += 1
            else:
                line = f"      · {raw}  → 前 3 是 {top_ids[:3]}"
                if top_ids:
                    line += f",top-1 属 {top1_family}"
                details.append(line)
                if top_ids and top1_family == own_family:
                    same_family.append(f"[{cid}] {raw} → {top_ids[0]}(同属 {own_family})")
                else:
                    cross_family.append(
                        f"[{cid}]({own_family}) {raw} → {top_ids[0]}({top1_family})")

            if not args.no_solve:
                if report.ok:
                    stat["solved"] += 1
                    totals["solved"] += 1
                elif getattr(report, "diverges", False):
                    # 定积分语料里,"正确判出发散"也是一个明确结论
                    stat["diverged"] += 1
                    totals["diverged"] += 1
                else:
                    stat["undecided"] += 1
                    totals["undecided"] += 1
                    details.append(f"      ✗ {raw}  → 既没算出值,也没判出发散")

        if stat["examples"] and stat["top1"] < stat["examples"]:
            problems.append(cid)

        bad = (stat["examples"] > 0 and
               (stat["parsed"] < stat["examples"] or stat["top1"] < stat["examples"]
                or stat["undecided"] > 0))
        if args.quiet and not bad:
            continue

        conclusion = stat["solved"] + stat["diverged"]
        print(f"{cid:<36}{stat['parsed']:>5}/{stat['examples']:<2}"
              f"{stat['top1']:>5}/{stat['examples']:<2}"
              f"{stat['top3']:>5}/{stat['examples']:<2}"
              f"{conclusion:>5}/{stat['examples']:<2}")
        for line in details:
            print(line)

    # ---------------- 汇总 ----------------
    print("-" * 78)
    n = max(totals["parsed"], 1)
    print(f"例题总数 {totals['examples']}  可解析 {totals['parsed']}"
          f"  检索 top1 自命中 {totals['top1']} ({totals['top1'] / n:.0%})"
          f"  检索 top3 自命中 {totals['top3']} ({totals['top3'] / n:.0%})")
    if not args.no_solve:
        print(f"算出有限值并通过验证 {totals['solved']}/{totals['parsed']}"
              f" ({totals['solved'] / n:.0%})")
        if args.definite:
            print(f"正确判定发散 {totals['diverged']}/{totals['parsed']}")
            print(f"既没算出值也没判出发散 {totals['undecided']}"
                  f" ({totals['undecided'] / n:.0%})")

    if same_family:
        print(f"\n○ 同族替代 {len(same_family)} 处 —— 方法类别相同,只是卡片分工重叠,可接受:")
        for line in same_family:
            print(f"    {line}")
    if cross_family:
        print(f"\n⚠ 跨族偏差 {len(cross_family)} 处 —— 推荐到了别的方法族,值得修:")
        for line in cross_family:
            print(f"    {line}")

    if problems:
        print(f"\n有 {len(problems)} 张卡片存在例题没能命中自己:"
              f"\n   {', '.join(problems)}")
        print("   先看上面的分类:同族替代不用管,跨族偏差才需要调 triggers 或换例题。")

    clean = (not errors and totals["parsed"] == totals["examples"]
             and totals["top1"] == totals["parsed"]
             and totals["undecided"] == 0)
    print("\n" + ("✅ 语料可用。" if clean else "⚠ 语料可用,但上面标出的地方值得修。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
