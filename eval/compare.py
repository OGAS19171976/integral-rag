"""两个评估报告之间的配对比较:结论到底站不站得住?

    python eval/compare.py A.json B.json
    python eval/compare.py A.json B.json --label-a "k=4" --label-b "k=8"
    python eval/compare.py A.json B.json --md eval/runs/report_k4_vs_k8.md

它**只读** run_eval.py 产出的 JSON(records 里带逐题的布尔结果),所以对任何
「同一套题、两个系统」的评估都通用 —— 这正是将来抽成 llm-eval-toolkit 的
report/ 模块要干的事。

结构上刻意分成两层:

    analyze_pair()  —— 纯函数,输入是两份 records,不碰文件、不打印;
    main()          —— 只负责读文件、打印、写 markdown。

这样分析逻辑可以被直接单元测试(不需要临时文件),而且抽库时搬走的
只有上面那一层。

为什么必须配对:

    同一个题目在 A、B 两版上的难度是一样的。把"题目难度"配掉之后,
    要检出的差异只来自系统改动本身。把它当成两组独立样本去比,
    等于把手上最贵的那部分信息扔了。

为什么必须多重比较校正:

    一次报 6 个指标、每个都用 alpha=0.05,那么"至少有一个假阳性"的概率
    接近 1 - 0.95^6 ≈ 26%。所以这里对同一批指标同时给
    Holm(强控制 FWER,对外下结论用)和 BH(控制 FDR,探索性分析用)。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 和 run_eval.py 同样的理由:Windows 控制台是 GBK。本脚本要把中文报告写进
# 管道或文件,不切编码就会直接抛 UnicodeEncodeError,或者输出一整片乱码 ——
# 一份没人能读的评估报告等于没做。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from evallib import stats as S   # noqa: E402

# 逐题布尔字段 -> 人话。只出现"两个报告里都有"的那些才会被采用。
CANDIDATE_METRICS: list[tuple[str, str]] = [
    ("ok", "求解 / 给出答案"),
    ("value_match", "与标注闭式一致"),
    ("family_hit", "top-1 方法族命中"),
    ("card_hit_top1", "top-1 卡片命中"),
    ("card_hit_top3", "top-3 卡片命中"),
    ("clean", "结果形式干净"),
]

WIDTH = 100

VERDICT_IDENTICAL = "逐题相同"
VERDICT_IMPROVED = "显著改善"
VERDICT_DEGRADED = "显著退化"
VERDICT_INCONCLUSIVE = "证据不足"


# ======================================================================
# 第一层:纯分析
# ======================================================================
@dataclass
class MetricComparison:
    key: str
    label: str
    n: int
    est_a: S.RateEstimate
    est_b: S.RateEstimate
    diff_ci: tuple[float, float]
    mcnemar: S.McNemarResult
    holm_p: float = float("nan")
    bh_p: float = float("nan")
    verdict: str = ""
    required_n: int = -1


@dataclass
class ComparisonResult:
    label_a: str
    label_b: str
    rows: list[MetricComparison] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    n_pairs: int = 0
    min_detectable_k: int = -1
    min_detectable_frac: float = float("nan")
    alpha: float = 0.05

    @property
    def comparable(self) -> bool:
        return bool(self.rows)


def parse_metric_override(text: str | None) -> list[tuple[str, str]] | None:
    if not text:
        return None
    known = dict(CANDIDATE_METRICS)
    return [(w.strip(), known.get(w.strip(), w.strip()))
            for w in text.split(",") if w.strip()]


def pick_metrics(recs_a: dict, recs_b: dict, override=None) -> list[tuple[str, str]]:
    """挑出可比指标:两边都在、且至少有一对非 None 的布尔值。"""
    if override:
        return override
    picked = []
    for key, label in CANDIDATE_METRICS:
        if paired_flags(recs_a, recs_b, key)[0]:
            picked.append((key, label))
    return picked


def paired_flags(recs_a: dict, recs_b: dict, key: str):
    """按题目 id 对齐,取两边都非 None 的项。返回 (ids, a, b)。

    注意 None 的语义:`value_match` 在"该题本来就该发散"时是 None,
    那不是"某一版失败",而是**这道题在这个指标上没有可比性**,
    必须整对丢掉,否则会把两版都算成错的。
    """
    ids, fa, fb = [], [], []
    for rid in recs_a:
        if rid not in recs_b:
            continue
        va, vb = recs_a[rid].get(key), recs_b[rid].get(key)
        if va is None or vb is None:
            continue
        ids.append(rid)
        fa.append(bool(va))
        fb.append(bool(vb))
    return ids, fa, fb


def analyze_pair(
    recs_a: dict,
    recs_b: dict,
    label_a: str = "A",
    label_b: str = "B",
    metrics: list[tuple[str, str]] | None = None,
    n_boot: int = 10_000,
    seed: int | None = 0,
    alpha: float = 0.05,
) -> ComparisonResult:
    """核心:两份 records → 一份带显著性判定的比较结果。纯函数。"""
    result = ComparisonResult(
        label_a=label_a, label_b=label_b, alpha=alpha,
        only_a=sorted(set(recs_a) - set(recs_b)),
        only_b=sorted(set(recs_b) - set(recs_a)),
    )

    for key, label in pick_metrics(recs_a, recs_b, metrics):
        ids, fa, fb = paired_flags(recs_a, recs_b, key)
        if not ids:
            continue
        result.rows.append(MetricComparison(
            key=key, label=label, n=len(ids),
            est_a=S.estimate_rate(label_a, fa, n_boot=n_boot, seed=seed),
            est_b=S.estimate_rate(label_b, fb, n_boot=n_boot, seed=seed),
            diff_ci=S.paired_bootstrap_diff_ci(
                [float(x) for x in fa], [float(x) for x in fb],
                n_boot=n_boot, alpha=alpha, seed=seed),
            mcnemar=S.mcnemar(fa, fb),
        ))

    if not result.rows:
        return result

    # 所有指标一起校正 —— 逐指标各自校正等于没校正
    holm_adj, holm_rej = S.holm_bonferroni(
        [r.mcnemar.exact_p for r in result.rows], alpha)
    bh_adj, _ = S.benjamini_hochberg(
        [r.mcnemar.exact_p for r in result.rows], alpha)

    for r, ha, hrej, ba in zip(result.rows, holm_adj, holm_rej, bh_adj):
        r.holm_p = float(ha)
        r.bh_p = float(ba)
        mc = r.mcnemar
        if mc.discordant == 0:
            r.verdict = VERDICT_IDENTICAL
        elif hrej:
            r.verdict = VERDICT_IMPROVED if mc.delta > 0 else VERDICT_DEGRADED
        else:
            r.verdict = VERDICT_INCONCLUSIVE
            if mc.n_pairs > 0:
                r.required_n = S.mcnemar_sample_size(
                    mc.b_only / mc.n_pairs, mc.a_only / mc.n_pairs, alpha, 0.8)

    result.n_pairs = max(r.n for r in result.rows)
    result.min_detectable_k, result.min_detectable_frac = \
        S.minimum_detectable_delta(result.n_pairs, alpha)
    return result


# ======================================================================
# 第二层:渲染
# ======================================================================
def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.1%}, {hi:+.1%}]"


def render_text(res: ComparisonResult, path_a: str = "", path_b: str = "",
                seconds_a=None, seconds_b=None) -> str:
    L: list[str] = []
    add = L.append
    add("=" * WIDTH)
    add(f"配对比较   {res.label_a}  →  {res.label_b}")
    add("=" * WIDTH)
    if path_a:
        add(f"  基线 {path_a}   累计耗时 {seconds_a} 秒")
        add(f"  候选 {path_b}   累计耗时 {seconds_b} 秒")
    if res.only_a:
        add(f"  ⚠ 只在基线里出现的题 {len(res.only_a)} 道,已排除:{res.only_a[:8]}"
            + (" ..." if len(res.only_a) > 8 else ""))
    if res.only_b:
        add(f"  ⚠ 只在候选里出现的题 {len(res.only_b)} 道,已排除:{res.only_b[:8]}"
            + (" ..." if len(res.only_b) > 8 else ""))
    if not res.comparable:
        add("  没有任何两边都可比的指标。")
        return "\n".join(L)

    add("")
    add("-" * WIDTH)
    add("表 1  比率与不确定性")
    add("-" * WIDTH)
    add(f"{'指标':<22}{res.label_a:>10}{'95% Wilson':>20}"
        f"{res.label_b:>10}{'95% Wilson':>20}")
    for r in res.rows:
        ea, eb = r.est_a, r.est_b
        add(f"{r.label:<22}"
            f"{ea.rate:>9.1%} "
            f"{f'[{ea.wilson[0]:.1%},{ea.wilson[1]:.1%}]':>20}"
            f"{eb.rate:>9.1%} "
            f"{f'[{eb.wilson[0]:.1%},{eb.wilson[1]:.1%}]':>20}")

    add("")
    add("-" * WIDTH)
    add("表 2  配对差异与显著性")
    add("-" * WIDTH)
    add(f"{'指标':<22}{'Δ':>9}{'95% CI (配对 Bootstrap)':>26}"
        f"{'分歧对':>8}{'净改善':>8}{'精确 p':>10}{'mid-p':>9}"
        f"{'Holm p':>9}{'结论':>14}")
    for r in res.rows:
        mc = r.mcnemar
        add(f"{r.label:<22}{mc.delta:>+8.1%} "
            f"{fmt_ci(*r.diff_ci):>25}"
            f"{mc.discordant:>8}{mc.a_only - mc.b_only:>+8}"
            f"{mc.exact_p:>10.4f}{mc.mid_p:>9.4f}{r.holm_p:>9.4f}"
            f"{r.verdict:>14}")

    add("")
    add("-" * WIDTH)
    add(f"检出能力(基于 {res.n_pairs} 道配对题)")
    add("-" * WIDTH)
    if res.min_detectable_k > 0:
        add(f"  净改善至少要 {res.min_detectable_k} 道题"
            f"({res.min_detectable_frac:.1%})才可能被判为显著"
            f"(最好情形:改进全部集中在一个方向)。")
    add("  注意:McNemar 以分歧对数量为条件,所以**单纯扩大评估集不会提高检出能力**,")
    add("        除非新补的题恰好是两版会分歧的那些。")

    add("")
    for r in res.rows:
        add(f"  · {r.label}")
        for line in r.mcnemar.render().splitlines():
            add(f"      {line}")
        if r.verdict == VERDICT_INCONCLUSIVE and r.required_n > 0:
            add(f"      按当前观测到的分歧率,要 80% 功效检出这个差异约需 "
                f"{r.required_n} 道配对题(现在只有 {r.mcnemar.n_pairs} 道)。")
            add("      —— 但这只是量级参考:分歧率本身是从这么小的样本估出来的。")
    return "\n".join(L)


def render_markdown(res: ComparisonResult, baseline_name: str,
                    candidate_name: str) -> str:
    lines = [
        f"# 配对比较:{res.label_a} → {res.label_b}",
        "",
        f"- 基线:`{baseline_name}`",
        f"- 候选:`{candidate_name}`",
        f"- 配对题数:{res.n_pairs}",
        f"- 显著性水平:α = {res.alpha}",
        "- 区间方法:Wilson(解析)+ 配对 Bootstrap(10000 次,种子 0,可复现)",
        "",
        "## 比率与不确定性",
        "",
        f"| 指标 | {res.label_a} | 95% Wilson | {res.label_b} | 95% Wilson "
        f"| Δ | 95% CI(配对) | 分歧对 | 精确 p | Holm p | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.rows:
        ea, eb, mc = r.est_a, r.est_b, r.mcnemar
        lines.append(
            f"| {r.label} "
            f"| {ea.rate:.1%} | [{ea.wilson[0]:.1%}, {ea.wilson[1]:.1%}] "
            f"| {eb.rate:.1%} | [{eb.wilson[0]:.1%}, {eb.wilson[1]:.1%}] "
            f"| {mc.delta:+.1%} | {fmt_ci(*r.diff_ci)} "
            f"| {mc.discordant} | {mc.exact_p:.4f} | {r.holm_p:.4f} "
            f"| {r.verdict} |")

    lines += ["", "## 检出能力", ""]
    if res.min_detectable_k > 0:
        lines.append(
            f"基于 {res.n_pairs} 道配对题,净改善至少要 **{res.min_detectable_k} 道题"
            f"({res.min_detectable_frac:.1%})** 才可能被判为显著"
            f"(最好情形:改进全部集中在一个方向)。")
    lines += [
        "",
        "> McNemar 精确检验以**分歧对的数量**为条件,因此单纯扩大评估集并不会"
        "提高检出能力 —— 除非新补的题目恰好是两版会分歧的那些。",
        "",
        "## 逐指标明细",
        "",
    ]
    for r in res.rows:
        lines += [f"### {r.label}", "", "```", r.mcnemar.render(), "```", ""]
        if r.verdict == VERDICT_INCONCLUSIVE and r.required_n > 0:
            lines += [
                f"按当前观测到的分歧率,要 80% 功效检出这个差异约需 "
                f"{r.required_n} 道配对题(现在 {r.mcnemar.n_pairs} 道)。"
                f"分歧率本身估计自小样本,这个数字只能当量级参考。", ""]
    return "\n".join(lines)


# ======================================================================
# 入口
# ======================================================================
def load_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "records" not in data:
        raise SystemExit(f"{path} 里没有 records —— 不是 run_eval.py 产出的报告")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="两个评估报告的配对比较")
    ap.add_argument("baseline", help="基线报告 JSON(改之前)")
    ap.add_argument("candidate", help="候选报告 JSON(改之后)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--metrics", default=None, help="逗号分隔,默认自动探测")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default=None, help="额外写一份 markdown 报告")
    args = ap.parse_args()

    path_a, path_b = Path(args.baseline), Path(args.candidate)
    rep_a, rep_b = load_report(path_a), load_report(path_b)
    label_a = args.label_a or path_a.stem
    label_b = args.label_b or path_b.stem

    res = analyze_pair(
        {r["id"]: r for r in rep_a["records"]},
        {r["id"]: r for r in rep_b["records"]},
        label_a=label_a, label_b=label_b,
        metrics=parse_metric_override(args.metrics),
        n_boot=args.n_boot, seed=args.seed, alpha=args.alpha,
    )
    print(render_text(res, str(path_a), str(path_b),
                      rep_a.get("seconds_total"), rep_b.get("seconds_total")))

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(res, path_a.name, path_b.name),
                       encoding="utf-8")
        print(f"\nmarkdown 报告已写入 {out}")
    return 0 if res.comparable else 1


if __name__ == "__main__":
    raise SystemExit(main())
