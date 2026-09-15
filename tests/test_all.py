"""端到端测试:三部分,全部通过才算健康。

    python tests/test_all.py

A. 解析层 —— LaTeX 各种写法能不能正确变成 SymPy 表达式
B. 验证层 —— 故意喂错的原函数,必须被抓住
C. 流水线 —— 一批教科书级积分能不能求解并通过验证

这个文件是从调试脚本演化来的,保留它的价值在于:
改动任何一层之后,一条命令就能知道有没有把别的东西弄坏。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integral_rag import analyze                                   # noqa: E402
from integral_rag.parse import (ParseError, parse_integral,          # noqa: E402
                               parse_integrand)
from integral_rag.retrieve import Retriever                         # noqa: E402
from integral_rag.verify import verify_antiderivative               # noqa: E402

CORPUS = ROOT / "corpus" / "method_cards.json"
DEFINITE_CORPUS = ROOT / "corpus" / "definite_cards.json"

FAILURES: list[str] = []


def check(condition: bool, label: str, extra: str = "") -> None:
    mark = "✓" if condition else "✗"
    print(f"  {mark} {label}" + (f"   {extra}" if extra else ""))
    if not condition:
        FAILURES.append(label)


# ================================================================ A. 解析层
# (输入, 期望的 SymPy 字符串)
PARSE_CASES = [
    (r"x^{3}", "x**3"),
    (r"\frac{1}{x}", "1/x"),
    (r"\sin(3x+1)", "sin(3*x + 1)"),
    (r"\sin 3x", "sin(3*x)"),
    (r"\sin x\cos x", "sin(x)*cos(x)"),
    (r"\sin^{3}x\cos^{2}x", "sin(x)**3*cos(x)**2"),
    (r"\sin^{2}x\cos^{2}x", "sin(x)**2*cos(x)**2"),
    (r"\sin 3x\cos 5x", "sin(3*x)*cos(5*x)"),
    (r"\tan^{3}x", "tan(x)**3"),
    (r"\tan^{3}x\sec x", "tan(x)**3*sec(x)"),
    (r"\sec^{4}x", "sec(x)**4"),
    (r"e^{x}\sin x", "exp(x)*sin(x)"),
    (r"x e^{x}", "x*exp(x)"),
    (r"x^{2}\sqrt{1-x^{2}}", "x**2*sqrt(1 - x**2)"),
    (r"\frac{1}{\sqrt{a^{2}-x^{2}}}", "1/sqrt(a**2 - x**2)"),
    (r"\frac{x^{2}}{\sqrt{1-x^{2}}}", "x**2/sqrt(1 - x**2)"),
    (r"\frac{1}{2+\cos x}", "1/(cos(x) + 2)"),
    (r"\frac{1}{1+e^{x}}", "1/(exp(x) + 1)"),
    (r"\frac{1}{e^{x}+e^{-x}}", "1/(exp(x) + exp(-x))"),
    (r"\frac{1}{x(x+1)^{2}}", "1/(x*(x + 1)**2)"),
    (r"x\sqrt{x+1}", "x*sqrt(x + 1)"),
    (r"\frac{1}{1+\sqrt{x+1}}", "1/(sqrt(x + 1) + 1)"),
    (r"\ln x", "log(x)"),
    (r"x\ln x", "x*log(x)"),
    (r"\arctan x", "atan(x)"),
    (r"\frac{\cos x}{\sin x}", "cos(x)/sin(x)"),
    (r"2x\sqrt{1+x^{2}}", "2*x*sqrt(x**2 + 1)"),
    (r"\frac{1}{1+x^{4}}", "1/(x**4 + 1)"),
    (r"1/(1+x^2)", "1/(x**2 + 1)"),
    (r"x**2*exp(x)", "x**2*exp(x)"),
    (r"\int \frac{x^{2}}{\sqrt{1-x^{2}}} dx", "x**2/sqrt(1 - x**2)"),
    (r"\int x^{3} dx", "x**3"),
    (r"\left(\frac{1}{x}\right)^{2}", "x**(-2)"),
]

# 这些输入必须被拒绝(而不是被悄悄解析成别的东西)
REJECT_CASES = [
    r"\int x^{2} dx + \unknowncommand{y}",
    r"x^{2}; import os",
    r"\frac{1}{x",
]


def test_parse() -> None:
    print("=" * 78)
    print("A. 解析层")
    print("=" * 78)
    for text, expected in PARSE_CASES:
        try:
            expr, _var = parse_integrand(text)
            actual = str(expr)
            check(actual == expected, f"{text!r} -> {expected}", f"实际 {actual!r}" if actual != expected else "")
        except ParseError as exc:
            check(False, f"{text!r} -> {expected}", f"解析失败:{str(exc).splitlines()[0]}")

    print("  必须拒绝的输入:")
    for text in REJECT_CASES:
        try:
            parse_integrand(text)
            check(False, f"{text!r} 应被拒绝", "却解析成功了")
        except ParseError:
            check(True, f"{text!r} 被正确拒绝")

    print("  语料里的全部例题:")
    cards = json.loads(CORPUS.read_text(encoding="utf-8"))
    ok = total = 0
    bad: list[str] = []
    for card in cards:
        for example in card.get("examples") or []:
            total += 1
            try:
                parse_integrand(example)
                ok += 1
            except ParseError as exc:
                bad.append(f"[{card.get('id')}] {example!r}: {str(exc).splitlines()[0]}")
    check(ok == total, f"{ok}/{total} 条例题可解析", "; ".join(bad[:3]))

    # ---- 特征 token:frullani_shape(§6i)
    # 这个 token 曾经是**死代码**:判据要求分子两项之和为零(即被积函数恒等于 0),
    # 而且取分子的方式(`sp.together` 后 fraction)还会把 exp 通分、
    # 让分母不再是 x。两处独立缺陷叠在一起,所以从来没触发过。
    print("  frullani_shape 特征判别:")
    from integral_rag.features import extract_definite
    frullani_cases = [
        (r"\int_{0}^{\infty} \frac{e^{-ax}-e^{-bx}}{x} dx", True),
        (r"\int_{0}^{\infty} \frac{\cos(ax)-\cos(bx)}{x} dx", True),
        (r"\int_{0}^{\infty} \frac{\arctan(3x)-\arctan(x)}{x} dx", True),
        (r"\int_{0}^{\infty} \frac{\sin x}{x} dx", False),          # Dirichlet,不是 Frullani
        (r"\int_{0}^{\infty} \frac{e^{-ax}+e^{-bx}}{x} dx", False),  # 相加不是相减
        (r"\int_{0}^{\infty} \frac{2e^{-x}-3e^{-2x}}{x} dx", False),  # 系数不等
        (r"\int_{0}^{\infty} \frac{e^{-(x+1)}-e^{-(x+2)}}{x} dx", False),  # f(ax+b)
    ]
    for text, expected in frullani_cases:
        parsed = parse_integral(text)
        feats = extract_definite(parsed.integrand, parsed.variable,
                                 parsed.lower, parsed.upper)
        got = "frullani_shape" in feats.misc
        check(got == expected, f"{text[:46]} frullani_shape={expected}",
              f"实际 {got}")


# ================================================================ B. 验证层
def test_verify() -> None:
    print()
    print("=" * 78)
    print("B. 验证层 —— 错的原函数必须被抓住")
    print("=" * 78)
    x = sp.Symbol("x", real=True)

    cases = [
        ("正确:sin x 是 cos x 的原函数", sp.sin(x), sp.cos(x), True),
        ("正确:x²/2 是 x 的原函数", sp.Rational(1, 2) * x**2, x, True),
        ("正确:arctan x 是 1/(1+x²) 的原函数", sp.atan(x), 1 / (1 + x**2), True),
        ("错误:sin x 拿去当 −sin x 的原函数", sp.sin(x), -sp.sin(x), False),
        ("错误:符号写反,−sin x 当 cos x 的原函数", -sp.sin(x), sp.cos(x), False),
        ("错误:系数错,2sin x 当 cos x 的原函数", 2 * sp.sin(x), sp.cos(x), False),
        ("错误:分支错,asin 当 −1/√(1−x²) 的原函数", sp.asin(x), -1 / sp.sqrt(1 - x**2), False),
        ("错误:未求值的 Integral 不算答案", sp.Integral(sp.sin(x), x), sp.sin(x), False),
    ]
    for label, F, target, expected in cases:
        result = verify_antiderivative(F, target, x)
        check(result.ok == expected, label,
              f"ok={result.ok}(期望 {expected}){'' if result.ok == expected else ' ' + result.detail}")


# ================================================================ C. 流水线
PIPELINE_CASES = [
    (r"\int x^{3} dx", True),
    (r"\int \sin(3x+1) dx", True),
    (r"\int x e^{x} dx", True),
    (r"\int \frac{x^{2}}{\sqrt{1-x^{2}}} dx", True),
    (r"\int \frac{1}{x^{2}-1} dx", True),
    (r"\int \frac{1}{2+\cos x} dx", True),
    (r"\int \frac{1}{1+e^{x}} dx", True),
    (r"\int \sin^{3}x\cos^{2}x dx", True),
    (r"\int x\sqrt{x+1} dx", True),
    (r"\int \frac{1}{1+x^{4}} dx", True),
    (r"\int e^{x}\sin x dx", True),
    (r"\int \frac{1}{\sqrt{x^{2}+4}} dx", True),
    (r"\int \ln x dx", True),
    (r"\int \arctan x dx", True),
    (r"\int \frac{\sqrt{x^{2}-a^{2}}}{x} dx", True),
    (r"\int \frac{1}{x\sqrt{x^{2}+x+1}} dx", True),
    # e^{x²} 没有初等原函数,但有特殊函数形式的原函数(√π·erfi(x)/2)。
    # 系统把它解出来并验证通过了 —— 这是合理的结果,不是幻觉。
    (r"\int e^{x^{2}} dx", True),
]

# 这些积分没有初等原函数,SymPy 也找不到特殊函数表达。
# 系统的正确行为是**拒答**,而不是编一个式子。所以期望是 False。
REFUSAL_CASES = [
    (r"\int x^{x} dx", False),
]

# 非初等、但能用特殊函数表达的积分:结果对不对不预设,
# 只强制检查一条不变量 —— **凡是被判定为"成功"的,必须已经通过验证**。
# 这条不变量正是整套系统的立身之本。
SPECIAL_FUNCTION_CASES = [
    r"\int \frac{\sin x}{x} dx",
    r"\int e^{e^{x}} dx",
    r"\int \sin(x^{2}) dx",
]


def test_pipeline() -> None:
    print()
    print("=" * 78)
    print("C. 流水线 —— 求解 + 验证")
    print("=" * 78)
    retriever = Retriever.from_json(CORPUS)

    def run(text: str):
        report = analyze(text, retriever=retriever, k=4)
        detail = str(report.result) if report.ok else ""
        if len(detail) > 60:
            detail = detail[:57] + "..."
        return report, detail

    for text, expected in PIPELINE_CASES + REFUSAL_CASES:
        try:
            report, detail = run(text)
        except Exception as exc:
            check(False, text, f"异常 {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        check(report.ok == expected, text, detail or "（按预期拒答）")

    print("  非初等(特殊函数)情形 —— 只检查「成功即已通过验证」这条不变量:")
    for text in SPECIAL_FUNCTION_CASES:
        try:
            report, detail = run(text)
        except Exception as exc:
            check(False, text, f"异常 {type(exc).__name__}: {exc}")
            continue
        if not report.ok:
            check(True, text, "按预期拒答")
            continue
        recheck = verify_antiderivative(report.result, report.f, report.x)
        check(recheck.ok, text, f"{detail}  [复核 {recheck.proof}]")

    # ---- 不定积分的解题步骤(§6h):换元链上必须有"回代"这一环
    # 定积分的换元是**等值变换**(不需要回代),不定积分**必须**回代 ——
    # 过去报告里缺的正是这一环(只给了一段"代换之后那个积分"的规则库路径)。
    chained = analyze(r"\int \frac{1}{2+\cos x} dx")      # 走万能代换
    chained_steps = chained.as_dict().get("steps") or []
    joined = " ".join(chained_steps)
    check("换元" in joined and "回代" in joined,
          "走代换的题,步骤里必须同时有「换元」与「回代」",
          f"实际 {chained_steps}")
    check(any("符号验证" in step for step in chained_steps),
          "步骤里要写明验证**凭什么**成立(哪一级化简把它变成 0)",
          f"实际 {chained_steps}")

    refused = analyze(r"\int x^{x} dx")                   # 拒答
    refused_steps = refused.as_dict().get("steps") or []
    check(not refused.ok and any("不作答" in step for step in refused_steps),
          "拒答也要给出结论性步骤,而不是一片空白",
          f"实际 {refused_steps}")


def main() -> int:
    test_parse()
    test_verify()
    test_pipeline()
    test_definite()
    test_quadrature()
    test_parametric()
    test_cas()
    test_kill()

    print()
    print("=" * 78)
    if FAILURES:
        print(f"❌ {len(FAILURES)} 项未通过:")
        for item in FAILURES:
            print(f"   · {item}")
        return 1
    print("✅ 全部通过")
    return 0


# ================================================================ D. 定积分
# (输入, 期望闭式;None 表示该题发散)
DEFINITE_CASES = [
    (r"\int_{0}^{1} x^{2} dx", "1/3"),
    (r"\int_{0}^{\pi} \sin x dx", "2"),
    (r"\int_{-1}^{1} x^{3} dx", "0"),
    (r"\int_{-a}^{a} x^{2} dx", "2*a**3/3"),
    (r"\int_{-1}^{2} |x| dx", "5/2"),
    (r"\int_{0}^{2\pi} |\sin x| dx", "4"),
    (r"\int_{0}^{1} \frac{1}{\sqrt{x}} dx", "2"),
    (r"\int_{0}^{1} \ln x dx", "-1"),
    (r"\int_{1}^{\infty} \frac{1}{x^{2}} dx", "1"),
    (r"\int_{0}^{\infty} e^{-x} dx", "1"),
    (r"\int_{-\infty}^{\infty} e^{-x^{2}} dx", "sqrt(pi)"),
    (r"\int_{0}^{\infty} \frac{1}{1+x^{2}} dx", "pi/2"),
    (r"\int_{0}^{\infty} \frac{\sin x}{x} dx", "pi/2"),
    (r"\int_{0}^{\infty} \cos(x^{2}) dx", "sqrt(2*pi)/4"),
    (r"\int_{0}^{\pi/2} \sin^{5}x dx", "8/15"),
    (r"\int_{0}^{\pi/2} \sin^{4}x dx", "3*pi/16"),
    (r"\int_{0}^{1} x^{2}(1-x)^{3} dx", "1/60"),
    (r"\int_{0}^{\infty} \frac{1}{\sqrt{x}(1+x)} dx", "pi"),
    (r"\int_{0}^{\pi} \frac{x\sin x}{1+\cos^{2}x} dx", "pi**2/4"),
    (r"\int_{0}^{\pi/2} \frac{\sin x}{\sin x+\cos x} dx", "pi/4"),
    (r"\int_{0}^{1} \frac{\ln(1+x)}{x} dx", "pi**2/12"),
    # 对数正弦族(§6e 新增的标准结果策略)。这一族 SymPy 全线放弃,
    # 值来自区间再现 + 倍角 + 半区间对称性的推导,所以单独立几道回归。
    (r"\int_{0}^{\pi/2} \ln(\sin x) dx", "-pi*log(2)/2"),
    (r"\int_{0}^{\pi/2} \ln(\cos x) dx", "-pi*log(2)/2"),
    (r"\int_{0}^{\pi} \ln(\sin x) dx", "-pi*log(2)"),
    (r"\int_{0}^{\pi/2} \ln(\sin x\cos x) dx", "-pi*log(2)"),
    # ln(tan) = ln(sin) − ln(cos),两半相消 ⇒ 0。
    # 这道题由**区间再现**策略而非对数正弦策略解出 —— 两条路线互相印证。
    (r"\int_{0}^{\pi/2} \ln(\tan x) dx", "0"),
    # Frullani 族(§6j 新增的标准结果策略)
    (r"\int_{0}^{\infty} \frac{e^{-x}-e^{-2x}}{x} dx", "log(2)"),
    (r"\int_{0}^{\infty} \frac{e^{-ax}-e^{-bx}}{x} dx", "log(b/a)"),
    (r"\int_{0}^{\infty} \frac{\arctan(ax)-\arctan(bx)}{x} dx", "pi*log(a/b)/2"),
    (r"\int_{1}^{\infty} \frac{1}{x} dx", None),
    (r"\int_{0}^{1} \frac{1}{x} dx", None),
    (r"\int_{0}^{\infty} \sin x dx", None),
    (r"\int_{0}^{\infty} \frac{\arctan x}{x} dx", None),   # 经典陷阱:看着像 π/2
]


def test_definite() -> None:
    print()
    print("=" * 78)
    print("D. 定积分/含参积分 —— 求解 + 数值验证")
    print("=" * 78)
    retriever = Retriever.from_json(DEFINITE_CORPUS, definite=True)
    symbols = {name: sp.Symbol(name, positive=True) for name in "abckmnpqrs"}
    for name in "xyzuvwt":
        symbols[name] = sp.Symbol(name, real=True)

    for text, expect in DEFINITE_CASES:
        try:
            report = analyze(text, retriever=retriever, k=4)
        except Exception as exc:
            check(False, text, f"异常 {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        if expect is None:
            check(getattr(report, "diverges", False), f"{text} 应判为发散",
                  f"实际 ok={report.ok} value={report.result}")
            continue

        if not report.ok:
            check(False, f"{text} 应求出 {expect}", "未通过验证")
            continue
        want = sp.sympify(expect, locals=symbols)
        # 参数集合要取"结果 ∪ 期望"两边的自由符号:参数可能只出现在**积分限**里
        # (∫_{-a}^{a} x²dx 的 a 就只在限里,从被积函数里找不到)。
        params = sorted({s for s in (report.result.free_symbols | want.free_symbols)
                         if s is not report.x and s.is_real}, key=lambda s: s.name)
        try:
            if params:
                subs = {p: sp.Float("0.5", 30) for p in params}
                got_value = complex(sp.N(report.result.subs(subs), 30))
                want_value = complex(sp.N(want.subs(subs), 30))
            else:
                got_value = complex(sp.N(report.result, 30))
                want_value = complex(sp.N(want, 30))
            rel = abs(got_value - want_value) / max(1e-30, abs(want_value))
            check(rel < 1e-6, f"{text} = {expect}",
                  f"实得 {report.result}(相对误差 {rel:.2e})")
        except Exception as exc:
            check(False, f"{text} 与期望值比对", f"{type(exc).__name__}: {exc}")

    # ---- 解题步骤的渲染(§6g):报告里必须看得出"这道题是怎么解出来的"
    diverged = analyze(r"\int_{1}^{\infty} \frac{1}{x} dx",
                       retriever=retriever, k=4)
    diverged_steps = " ".join(diverged.as_dict().get("steps") or [])
    check("发散" in diverged_steps and "结论" in diverged_steps,
          "发散题的步骤里应写明判据与结论",
          f"实际步骤:{diverged_steps[:90]}")

    solved = analyze(r"\int_{0}^{\pi/2} \ln(\sin x) dx", retriever=retriever, k=4)
    solved_steps = solved.as_dict().get("steps") or []
    check(len(solved_steps) >= 3
          and any("求解" in step for step in solved_steps)
          and any("验证" in step for step in solved_steps),
          "解出来的题应给出「求解 → 验证」的步骤",
          f"实际 {solved_steps}")

    # ---- Frullani 标准结果策略(§6j):arctan 型 SymPy 积不出来,
    # 必须由这条策略接手(以前只能靠费曼技巧绕,要 20 秒以上)。
    frullani = analyze(r"\int_{0}^{\infty} \frac{\arctan(ax)-\arctan(bx)}{x} dx",
                       retriever=retriever, k=4)
    check(frullani.ok and "Frullani" in (frullani.solution.strategy or ""),
          "arctan 型 Frullani 积分应由 Frullani 标准结果策略解出",
          f"实际 strategy={frullani.solution.strategy!r}")


# ================================================================ E. 数值层
# 这些正是"只用 mpmath.quad 会翻车"的对抗性例子。
# 数值层如果不把它们摆平,验证器就会给出假阳性或假阴性。
QUADRATURE_CASES = [
    # (被积函数, 下, 上, 期望值;None 表示发散)
    (lambda x: sp.Abs(x), -1, 2, sp.Rational(5, 2), "折点必须切开"),
    (lambda x: 1 / sp.sqrt(x), 0, 1, sp.Integer(2), "端点奇点 tanh-sinh"),
    (lambda x: 1 / x, 1, sp.oo, None, "对数发散必须判出来"),
    (lambda x: 1 / x, 0, 1, None, "一阶极点不能报成有限值"),
    (lambda x: sp.sin(x), 0, sp.oo, None, "不衰减振荡必发散"),
    (lambda x: sp.sin(x) / x, 0, sp.oo, sp.pi / 2, "条件收敛"),
    (lambda x: sp.cos(x**2), 0, sp.oo, sp.sqrt(2 * sp.pi) / 4, "加速振荡"),
    (lambda x: sp.exp(-x**2), -sp.oo, sp.oo, sp.sqrt(sp.pi), "双侧无穷"),
]


def test_quadrature() -> None:
    print()
    print("=" * 78)
    print("E. 数值层 —— 对抗性检验(只用 mpmath.quad 会翻车的那些)")
    print("=" * 78)
    from integral_rag.quadrature import evaluate
    x = sp.Symbol("x", real=True)
    for build, lo, hi, expect, note in QUADRATURE_CASES:
        f = build(x)
        result = evaluate(f, x, lo, hi)
        if expect is None:
            check(result.divergent, f"{note}:应判发散",
                  f"实际 {result.describe()}")
            continue
        if not result.ok:
            check(False, f"{note}:应得 {sp.N(expect, 10)}", f"实际 {result.describe()}")
            continue
        rel = abs(complex(result.value) - complex(sp.N(expect, 30))) / max(
            1e-30, abs(complex(sp.N(expect, 30))))
        check(rel < 1e-7, f"{note}:{sp.N(expect, 10)}",
              f"实得 {result.describe()}(相对误差 {rel:.2e})")


# ================================================================ F. 含参积分
def test_parametric() -> None:
    print()
    print("=" * 78)
    print("F. 含参积分 —— 积分号下求导的数值验证")
    print("=" * 78)
    from integral_rag.parametric import verify_derivative_under_integral
    x = sp.Symbol("x", real=True)
    t = sp.Symbol("t", positive=True)

    # 合法的交换:∫_0^∞ e^{−tx}dx,两边都成立
    good = verify_derivative_under_integral(sp.exp(-t * x), x, t, 0, sp.oo)
    check(bool(good) and all(c.agree for c in good),
          "∫_0^∞ e^{-tx}dx:积分号下求导应被判为成立",
          "; ".join(c.note for c in good))

    # 不合法的交换:右边 ∫cos(tx)dx 发散 —— 数值层应当**否证**它
    bad = verify_derivative_under_integral(sp.sin(t * x) / x, x, t, 0, sp.oo)
    check(bool(bad) and not any(c.agree for c in bad),
          "∫_0^∞ sin(tx)/x dx:积分号下求导应被判为不成立",
          "; ".join(c.note for c in bad))

    # 费曼技巧要能真的算出闭式
    from integral_rag.parametric import feynman
    a, b = sp.symbols("a b", positive=True)
    results = feynman((sp.atan(a * x) - sp.atan(b * x)) / x, x, 0, sp.oo)
    check(bool(results), "费曼技巧:∫(arctan ax − arctan bx)/x dx 应给出闭式",
          str(results[0].value) if results else "没有结果")


# ================================================================ G. CAS 闸口
def test_cas() -> None:
    """cas.py 的三条保证:记忆化、超时不再重试、泄漏到上限就落闸。

    这些是为**性能**做的修复,但性能修复同样需要回归测试。
    否则后人一个手滑把记忆化关掉,正确性测试会全绿,什么都发现不了 ——
    而评估耗时悄悄涨回 2.4 倍。
    """
    print()
    print("=" * 78)
    print("G. CAS 闸口 —— 记忆化 / 超时记忆 / 泄漏闸门")
    print("=" * 78)
    from integral_rag import cas

    cas.reset()

    # 1) 记忆化:同一个键只真正算一次
    calls: list = []

    def counted(marker):
        calls.append(marker)
        return marker

    status, value = cas.guard("k1", lambda: counted(42), 5.0)
    status2, value2 = cas.guard("k1", lambda: counted(99), 5.0)
    check(status == "ok" and value == 42 and value2 == 42 and len(calls) == 1,
          "同一个键只算一次,后续直接命中记忆",
          f"status={status}/{status2} value={value}/{value2} "
          f"实际调用 {len(calls)} 次")

    # 2) 超时记忆:这是主要的时间节省点 —— 超时过的调用不再发起第二次
    slow: list = []

    def sleepy():
        slow.append(1)
        time.sleep(1.0)

    s1, _ = cas.guard("k2", sleepy, 0.05)
    s2, _ = cas.guard("k2", sleepy, 0.05)
    check(s1 == "timeout" and s2 == "timeout" and len(slow) == 1,
          "超时过的调用不会重复发起",
          f"status={s1}/{s2} 实际发起 {len(slow)} 次")

    # 3) 泄漏计量:只做观察,不拦截任何调用。
    #    闸门(按僵尸数拒绝/压缩长调用)做过两版,两版都造成正确性回归 ——
    #    判据是"请求了多长预算"而非"实际要跑多久",必然误伤
    #    "请求 12 秒、实际 1 毫秒"的调用,还会让前后题目预算互相耦合。
    #    所以这里只验证它**数得准**,不验证它拦得住。
    cas.reset()
    for index in range(cas.MAX_RUNAWAYS):
        cas.guard(f"hog{index}", lambda: time.sleep(2.0), 0.05)
    count = cas.runaway_count()
    check(count == cas.MAX_RUNAWAYS,
          f"泄漏计量能数出 {cas.MAX_RUNAWAYS} 个跑飞线程",
          f"实际数出 {count} 个")

    s3, v3 = cas.guard("k3", lambda: 7, cas.LONG_CALL + 1.0)
    check(s3 == "ok" and v3 == 7,
          "有僵尸线程在跑也不影响新调用(闸门已拆,不做拦截)",
          f"实际 status={s3} value={v3}")

    cas.reset()


# ================================================================ H. 超时清理
def test_kill() -> None:
    """H. 超时之后那个跑飞的计算**是否真的停了**。

    这是"结构性"的一条。超时只放弃等待、不停止计算时,线程会继续占着
    CPU 和 GIL,把后面每一道题都拖慢(README §6c.1)。现在超时会往
    线程**家族**里注入异步异常把它们打断。

    ## 判据是 CPU,不是线程死活

    一开始我用"线程还在不在"当判据,结果不稳定 —— 因为外层线程在内层被
    杀之后会**恢复并继续干自己的活**(还在自己的预算里收尾),它是活着的,
    但并没有在烧 CPU。真正要证明的是"**不再有跑飞的计算占着核**",
    所以直接量进程 CPU 增量:对照组(放着不管)应该吃满一个核,
    实验组(超时处理过)应该接近 0。
    """
    print()
    print("=" * 78)
    print("H. 超时清理 —— 跑飞的计算有没有真的停下(按 CPU 增量判定)")
    print("=" * 78)
    import threading

    from integral_rag.timeout import _kill_thread, run_with_timeout

    x = sp.Symbol("x")

    def slow():
        try:
            sp.integrate(sp.log(1 + x) / (1 + x ** 2), x)   # 已知会陷进去很久
        except BaseException:                              # noqa: BLE001
            pass          # 对照组会被主动杀掉,别让它往 stderr 喷栈

    def cpu_over(seconds: float) -> float:
        start = time.process_time()
        time.sleep(seconds)
        return time.process_time() - start

    # ---- 对照组:跑飞的计算放着不管,量它吃多少 CPU
    burner = threading.Thread(target=slow, daemon=True)
    burner.start()
    time.sleep(0.3)
    uncontrolled = cpu_over(1.2)
    _kill_thread(burner)
    burner.join(2)

    check(uncontrolled > 0.5,
          "对照组确实在烧 CPU(说明这个测量本身有效)",
          f"1.2 秒内只烧了 {uncontrolled:.2f}s")

    # ---- 实验组:同样的计算交给超时处理
    status, _ = run_with_timeout(slow, 1.0)
    controlled = cpu_over(1.2)

    check(status == "timeout", "跑飞的积分应被超时拦下", f"实际 status={status}")
    check(controlled < uncontrolled * 0.25,
          "超时之后跑飞的计算被停下(不再继续烧 CPU)",
          f"对照组 {uncontrolled:.2f}s vs 实验组 {controlled:.2f}s")

    # ---- 关键安全性质:杀完之后进程仍然可用(没有泄漏的锁/坏掉的缓存)
    recovered = sp.integrate(sp.exp(-x ** 2), x)
    check(recovered is not None and not recovered.has(sp.Integral),
          "杀掉跑飞线程之后符号计算仍然正常(没有泄漏的锁/坏掉的缓存)",
          f"实际 {recovered}")
    check(sp.simplify(sp.diff(sp.sin(x) * sp.cos(x), x) - sp.cos(2 * x)) == 0,
          "杀掉之后 diff/simplify 结果仍然正确",
          "化简结果不为 0")

    # ---- 而且不是一次性的
    status2, _ = run_with_timeout(slow, 1.0)
    check(status2 == "timeout", "连续超时都能被清理(不是一次性的)",
          f"实际 status={status2}")


if __name__ == "__main__":
    raise SystemExit(main())
