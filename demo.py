"""命令行演示入口。

    python demo.py                      # 跑一组展示题(第一题出完整报告,其余出汇总表)
    python demo.py "\\int \\frac{1}{2+\\cos x} dx"   # 单题完整报告
    python demo.py -i                   # 交互模式
    python demo.py --method "根号里面是平方差"        # 按中文描述查方法卡片
    python demo.py -b "\\int x^2 dx"      # 精简报告(省略方法指导与尝试记录)
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from integral_rag import analyze, find_method                      # noqa: E402
from integral_rag.parse import ParseError                          # noqa: E402
from integral_rag.retrieve import Retriever                        # noqa: E402

CORPUS = str(Path(__file__).resolve().parent / "corpus" / "method_cards.json")
DEFINITE_CORPUS = str(Path(__file__).resolve().parent / "corpus" / "definite_cards.json")


def _display_width(text: str) -> int:
    """终端里的显示宽度。中文是双宽字符,直接 ljust 会算错、表格会歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))

# 每题覆盖一个大类,顺序按教材讲授顺序排
SHOWCASE = [
    ("直接套基本积分表", r"\int x^{3} dx"),
    ("凑微分(内层是线性函数)", r"\int \sin(3x+1) dx"),
    ("凑微分(把 g'(x)dx 凑进微分)", r"\int x e^{x^{2}} dx"),
    ("分部积分(多项式×指数)", r"\int x e^{x} dx"),
    ("分部积分(单独的 ln x)", r"\int \ln x dx"),
    ("循环分部", r"\int e^{x}\sin x dx"),
    ("有理函数:部分分式", r"\int \frac{1}{x^{2}-1} dx"),
    ("有理函数:重根", r"\int \frac{1}{x(x+1)^{2}} dx"),
    ("一次根式代换", r"\int x\sqrt{x+1} dx"),
    ("三角替换 √(a²−x²)", r"\int \frac{x^{2}}{\sqrt{1-x^{2}}} dx"),
    ("三角替换 √(x²+a²)", r"\int \frac{1}{\sqrt{x^{2}+4}} dx"),
    ("三角幂次:拆一个凑微分", r"\int \sin^{3}x\cos^{2}x dx"),
    ("三角有理式:万能代换", r"\int \frac{1}{2+\cos x} dx"),
    ("指数有理式:t = eˣ", r"\int \frac{1}{1+e^{x}} dx"),
    ("配对技巧(1/(1+x⁴))", r"\int \frac{1}{1+x^{4}} dx"),
]

# 定积分题库。这一组重点展示"数值求积当独立真值"和"发散判定"。
DEFINITE_SHOWCASE = [
    ("正常积分", r"\int_{0}^{1} x^{2} dx"),
    ("对称区间上的奇函数", r"\int_{-1}^{1} x^{3} dx"),
    ("含折点(先切开再积)", r"\int_{-1}^{2} |x| dx"),
    ("端点奇点(可积)", r"\int_{0}^{1} \frac{1}{\sqrt{x}} dx"),
    ("无穷限", r"\int_{0}^{\infty} e^{-x} dx"),
    ("Dirichlet 积分", r"\int_{0}^{\infty} \frac{\sin x}{x} dx"),
    ("Fresnel 积分", r"\int_{0}^{\infty} \cos(x^{2}) dx"),
    ("Wallis 公式", r"\int_{0}^{\pi/2} \sin^{5}x dx"),
    ("区间再现公式", r"\int_{0}^{\pi} \frac{x\sin x}{1+\cos^{2}x} dx"),
    ("发散判定(不是算不出来)", r"\int_{1}^{\infty} \frac{1}{x} dx"),
    ("发散陷阱:看着像 π/2", r"\int_{0}^{\infty} \frac{\arctan x}{x} dx"),
    ("含参:参数范围是答案的一部分", r"\int_{0}^{\infty} x^{s-1} e^{-x} dx"),
    ("含参:必须靠费曼技巧", r"\int_{0}^{\infty} \frac{\arctan(ax)-\arctan(bx)}{x} dx"),
]


def one(text: str, retriever, brief: bool, k: int) -> bool:
    try:
        report = analyze(text, retriever=retriever, k=k)
    except ParseError as exc:
        print(f"\n[解析失败] {exc}\n")
        return False
    print(report.render(brief=brief))
    return report.ok


def showcase(retriever, brief: bool, k: int, cases=None, title: str = "") -> None:
    cases = cases or SHOWCASE
    print(f"{title}共 {len(cases)} 题。先看第一题的完整报告,再看汇总表。\n")
    first_title, first_text = cases[0]
    print(f"### {first_title}")
    one(first_text, retriever, brief=False, k=k)

    print("=" * 100)
    print("其余各题汇总(完整报告请把题目作为参数单独跑)")
    print("=" * 100)
    print(_pad("类别", 34) + _pad("top-1 方法卡片", 32) + "结果")
    print("-" * 100)
    ok = 0
    diverged = 0
    for label, text in cases:
        try:
            report = analyze(text, retriever=retriever, k=k)
        except ParseError as exc:
            print(_pad(label, 34) + _pad("解析失败", 32) + str(exc))
            continue
        if report.ok:
            ok += 1
            result = str(report.result)
            if _display_width(result) > 44:
                result = result[:41] + "..."
            extra = ""
            conditions = list(getattr(report.solution, "conditions", []))
            if conditions:
                extra = f"  ⚠ {conditions[0][:34]}"
            print(_pad(label, 34) + _pad(report.top1_card_id or "-", 32) + result + extra)
        elif getattr(report, "diverges", False):
            diverged += 1
            print(_pad(label, 34) + _pad(report.top1_card_id or "-", 32)
                  + "⛔ 发散(已确认,不是算不出来)")
        else:
            numeric = getattr(report.solution, "numeric", None)
            note = f"数值层:{numeric.describe()}" if numeric else ""
            print(_pad(label, 34) + _pad(report.top1_card_id or "-", 32)
                  + f"未通过验证  {note}")
    print("-" * 100)
    # 判出发散也是**正确的结论**,不该和"没做出来"混在一起计数
    print(f"合计:{ok}/{len(cases)} 题给出了通过验证的结果"
          f"(另有 {diverged} 题正确判定为发散)")


def interactive(retriever, brief: bool, k: int) -> None:
    print("输入积分(LaTeX 或纯文本都行),直接回车退出。")
    print(r"例如:\int \frac{x^{2}}{\sqrt{1-x^{2}}} dx   或   x*e^x")
    print()
    while True:
        try:
            text = input("∫ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        if "?" in text or "怎么" in text or "怎么做" in text:
            for hit in find_method(text, k=k, retriever=retriever):
                print(f"  [{hit.score:5.2f}] {hit.id} — {hit.name}")
            print()
            continue
        one(text, retriever, brief, k)


def method_lookup(description: str, retriever, k: int) -> None:
    hits = find_method(description, k=k, retriever=retriever)
    print(f"按描述检索方法卡片:{description!r}\n")
    for rank, hit in enumerate(hits, start=1):
        card = hit.card
        print(f"{rank}. [{hit.score:5.2f}] {hit.id} — {card.get('name')}")
        print(f"   适用: {card.get('when')}")
        recipe = str(card.get("recipe", "")).replace("\n", "\n         ")
        print(f"   做法: {recipe}")
        print(f"   易错: {card.get('pitfalls')}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="一元积分:检索 + CAS + 验证(不定 / 定 / 含参)")
    parser.add_argument("expr", nargs="?", help="要计算的积分,LaTeX 或纯文本")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("-b", "--brief", action="store_true", help="精简报告")
    parser.add_argument("-k", type=int, default=4, help="检索取前 k 张卡片(默认 4)")
    parser.add_argument("--method", help="按中文描述检索方法卡片(不定积分语料)")
    parser.add_argument("--definite", action="store_true",
                        help="跑定积分/含参积分展示题库")
    parser.add_argument("--corpus", default=None, help="方法卡片语料路径")
    args = parser.parse_args()

    corpus = args.corpus or (DEFINITE_CORPUS if args.definite else CORPUS)
    try:
        retriever = Retriever.from_json(corpus, definite=bool(args.definite))
    except Exception as exc:
        print(f"[错误] 加载语料失败:{exc}")
        return 2

    kind = "定积分/含参积分" if args.definite else "不定积分"
    print(f"{kind}方法卡片语料:{corpus}  ({len(retriever.cards)} 张卡片)")
    print()

    if args.method:
        method_lookup(args.method, retriever, args.k)
        return 0
    if args.expr:
        return 0 if one(args.expr, retriever, args.brief, args.k) else 1
    if args.interactive:
        interactive(retriever, args.brief, args.k)
        return 0

    if args.definite:
        showcase(retriever, args.brief, args.k, DEFINITE_SHOWCASE, "定积分展示题库")
    else:
        showcase(retriever, args.brief, args.k, SHOWCASE, "不定积分展示题库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
