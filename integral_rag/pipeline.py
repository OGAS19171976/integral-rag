"""主流水线:解析 → 结构特征 → 检索 → 规划 → 执行 → 验证 → 报告。

一条设计原则贯穿始终:
    **任何进入最终答案的东西,都必须先过验证层。**
    检索到的方法只是"建议",CAS 的结果只是"候选",只有验证器点头了,
    才算答案。所以这个系统可以坦率地说"我没做出来",而不会编一个式子糊弄你。

命令行入口见项目根目录的 demo.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp

from .features import Features, extract, extract_definite
from .parse import IntegralInput, ParseError, parse_integral
from .plan import Plan, describe_plan, make_plan
from .retrieve import DEFAULT_CORPUS, Hit, Retriever, describe_hits
from .solve import Solution, solve
from .timeout import run_with_timeout

# 含参专项检查的时间上限(它们只是增值功能)
PARAMETRIC_TIMEOUT = 15.0

# 两项含参附加分析(积分号下求导 + 尾部一致性探查)**合计**的时间上限。
# 原来是各给 PARAMETRIC_TIMEOUT、串行两段,于是一道含参题允许在
# "增值分析"上烧掉 30 秒 —— 而这段开销对主结果毫无贡献。
# 实测 ∫_0^∞ e^{-t x²}dx 独立跑 26.3 秒,绝大部分正是花在这里。
SUPPLEMENTARY_BUDGET = 5.0

_RETRIEVER: Retriever | None = None
_RETRIEVER_PATH: Path | None = None
_DEFINITE_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "definite_cards.json"
_DEFINITE_RETRIEVER: Retriever | None = None
_DEFINITE_RETRIEVER_PATH: Path | None = None

LINE = "─" * 62


def get_retriever(path: str | Path | None = None) -> Retriever:
    """按需加载方法卡片语料,进程内缓存。"""
    global _RETRIEVER, _RETRIEVER_PATH
    target = Path(path) if path else DEFAULT_CORPUS
    if _RETRIEVER is None or _RETRIEVER_PATH != target:
        _RETRIEVER = Retriever.from_json(target)
        _RETRIEVER_PATH = target
    return _RETRIEVER


def get_definite_retriever(path: str | Path | None = None) -> Retriever:
    """定积分/含参积分的方法卡片语料。"""
    global _DEFINITE_RETRIEVER, _DEFINITE_RETRIEVER_PATH
    target = Path(path) if path else _DEFINITE_CORPUS
    if _DEFINITE_RETRIEVER is None or _DEFINITE_RETRIEVER_PATH != target:
        _DEFINITE_RETRIEVER = Retriever.from_json(target, definite=True)
        _DEFINITE_RETRIEVER_PATH = target
    return _DEFINITE_RETRIEVER


def _short(expr, limit: int = 78) -> str:
    text = str(expr)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class Report:
    input_text: str
    f: sp.Expr
    x: sp.Symbol
    features: Features
    hits: list[Hit]
    plan: Plan
    solution: Solution

    # ---------------------------------------------------------- 便捷属性
    @property
    def ok(self) -> bool:
        return self.solution.ok

    @property
    def result(self):
        return self.solution.result

    @property
    def result_latex(self) -> str:
        if self.solution.result is None:
            return ""
        try:
            return sp.latex(self.solution.result)
        except Exception:
            return str(self.solution.result)

    @property
    def primary_card(self) -> dict | None:
        return self.hits[0].card if self.hits else None

    @property
    def top1_card_id(self) -> str | None:
        return self.hits[0].id if self.hits else None

    # ---------------------------------------------------------- 渲染
    def render(self, brief: bool = False) -> str:
        out: list[str] = []
        out.append(LINE)
        out.append(f"输入: {self.input_text}")
        out.append(f"被积函数: {self.f}")
        out.append(f"积分变量: {self.x}")
        out.append(LINE)

        out.append("① 结构特征")
        out.append(f"   {self.features.describe()}")
        out.append("")

        out.append(f"② 检索命中(方法卡片 top-{len(self.hits)})")
        out.append(describe_hits(self.hits))
        out.append("")

        if not brief:
            card = self.primary_card
            if card:
                out.append(f"③ 方法指导(来自 top-1 卡片 {card.get('id')})")
                out.append(f"   名称: {card.get('name')}")
                out.append(f"   适用: {card.get('when')}")
                recipe = str(card.get("recipe", "")).replace("\n", "\n         ")
                out.append(f"   做法: {recipe}")
                pitfalls = str(card.get("pitfalls", "")).replace("\n", "\n         ")
                out.append(f"   易错: {pitfalls}")
                out.append("")

            out.append("④ 执行计划")
            out.append(describe_plan(self.plan))
            out.append("")

        out.append("⑤ 结果与验证")
        if self.solution.ok:
            out.append(f"   ∫ {_short(self.f, 60)} dx = {_short(self.result)}")
            out.append(f"   LaTeX: {self.result_latex}")
            out.append(f"   ✅ 已验证 · 证据链:{self.solution.proof}")
            if self.solution.detail:
                out.append(f"   ⚠ {self.solution.detail}")
        else:
            out.append("   ❌ 未能给出通过验证的结果")
            out.append("   ❌ 未能给出通过验证的结果")
            out.append("   这不是「没找到」,而是「没算出来」——")
            out.append("   系统选择不作答,而不是编一个看起来像样的原函数给你。")
        out.append("")

        if not brief:
            # ★ 解题步骤:把"原积分 → 换元 → 求原函数 → 回代 → 验证"串成一条链。
            #   以前这里只有一段 SymPy 规则库给的路径,而那段路径描述的是
            #   **代换之后那个积分**的方法 —— 看得出方法,看不出它怎么接回原积分。
            steps = list(getattr(self.solution, "steps", []) or [])
            if steps:
                out.append("⑥ 解题步骤")
                for index, step in enumerate(steps, start=1):
                    out.append(f"   第 {index} 步 · {step}")
                out.append("")

            if self.solution.trace:
                var = self.solution.trace_var
                if steps:
                    # 有步骤时,规则库路径是**挂在**"求原函数"那一步下面的细节,
                    # 所以要说明它讲的是哪个积分 —— 只有真走了代换才谈得上
                    # "代换后的变量"。
                    where = ("即代换后的积分变量"
                             if "代换" in (self.solution.strategy or "")
                             else "与原积分同一个变量")
                    out.append(f"   ↳ 其中「求原函数」那一步,SymPy 规则库展开如下"
                               f"(变量 {var},{where}):")
                else:
                    out.append(f"⑥ 方法链(SymPy 逐步积分器给出的解题路径,变量 {var})")
                for step in self.solution.trace:
                    pad = "   " + "  " * step["depth"]
                    detail = f"  {step['detail']}" if step["detail"] else ""
                    out.append(f"{pad}{step['label']}{detail}")
                out.append("")
            elif self.solution.ok and not steps:
                out.append("⑥ 方法链")
                out.append("   SymPy 的规则库没给出逐步路径(它直接用了内置算法)。")
                out.append("")

            out.append(f"⑦ 尝试记录(共 {len(self.solution.attempts)} 次)")
            for attempt in self.solution.attempts:
                if attempt.result is not None and attempt.verify is not None and attempt.verify.ok:
                    mark = "✅"
                    tail = f"→ {_short(attempt.result, 52)}  [{attempt.verify.proof}]"
                elif attempt.result is not None and attempt.verify is not None:
                    mark = "❌"
                    tail = f"→ 结果被验证层否决:{attempt.verify.detail}"
                else:
                    mark = "…"
                    tail = f"→ {attempt.note or '未得到结果'}"
                out.append(f"   {mark} {attempt.strategy}")
                out.append(f"        {tail}")
            for err in self.solution.errors:
                out.append(f"   ⚠ {err}")
            out.append("")

        return "\n".join(out)

    def as_dict(self) -> dict:
        """结构化输出,供评估脚本和上层程序使用。"""
        return {
            "input": self.input_text,
            "integrand": str(self.f),
            "variable": str(self.x),
            "features": {
                "families": sorted(self.features.families),
                "forms": sorted(self.features.forms),
                "radicals": sorted(self.features.radicals),
                "misc": sorted(self.features.misc),
            },
            "top1_card": self.top1_card_id,
            "top_hits": [{"id": h.id, "score": round(h.score, 4),
                          "example_sim": round(h.example_sim, 4)} for h in self.hits],
            "strategy": self.solution.strategy,
            "result": str(self.solution.result) if self.solution.result is not None else None,
            "result_latex": self.result_latex,
            "ok": self.solution.ok,
            "proof": self.solution.proof,
            "steps": list(getattr(self.solution, "steps", []) or []),
            "n_attempts": len(self.solution.attempts),
        }


# ---------------------------------------------------------------- 定积分报告
@dataclass
class DefiniteReport:
    """定积分/含参积分的完整报告。属性名和不定积分的 Report 对齐,
    这样上层代码(评估脚本等)可以不加区分地用 `.ok` / `.result` / `.render()`。
    """

    input_text: str
    f: sp.Expr
    x: sp.Symbol
    lower: sp.Expr
    upper: sp.Expr
    features: Features
    hits: list[Hit]
    plan: Plan
    solution: "object"                       # DefiniteSolution(避免循环导入)
    derivative_checks: list = field(default_factory=list)
    uniform_note: str = ""

    @property
    def ok(self) -> bool:
        return bool(getattr(self.solution, "ok", False))

    @property
    def result(self):
        return getattr(self.solution, "value", None)

    @property
    def result_latex(self) -> str:
        return getattr(self.solution, "value_latex", "")

    @property
    def diverges(self) -> bool:
        return bool(getattr(self.solution, "diverges", False))

    @property
    def primary_card(self) -> dict | None:
        return self.hits[0].card if self.hits else None

    @property
    def top1_card_id(self) -> str | None:
        return self.hits[0].id if self.hits else None

    @property
    def is_parametric(self) -> bool:
        return "parameterized" in self.features.misc

    def render(self, brief: bool = False) -> str:
        out: list[str] = []
        # 小节编号按**实际渲染顺序**生成,而不是写死 ①②③…
        # 写死的话 brief 模式(跳过某些小节)会跳号,以后加小节也会错位。
        marks = iter("①②③④⑤⑥⑦⑧⑨")

        def mark(title: str) -> str:
            return f"{next(marks)} {title}"

        out.append(LINE)
        out.append(f"输入: {self.input_text}")
        out.append(f"被积函数: {self.f}")
        out.append(f"积分区间: [{self.lower}, {self.upper}]  积分变量: {self.x}")
        out.append(LINE)

        out.append(mark("结构特征"))
        out.append(f"   {self.features.describe()}")
        out.append("")

        out.append(mark(f"检索命中(定积分方法卡片 top-{len(self.hits)})"))
        out.append(describe_hits(self.hits))
        out.append("")

        if not brief:
            card = self.primary_card
            if card:
                out.append(mark(f"方法指导(来自 top-1 卡片 {card.get('id')})"))
                out.append(f"   名称: {card.get('name')}")
                out.append(f"   适用: {card.get('when')}")
                recipe = str(card.get("recipe", "")).replace("\n", "\n         ")
                out.append(f"   做法: {recipe}")
                pitfalls = str(card.get("pitfalls", "")).replace("\n", "\n         ")
                out.append(f"   易错: {pitfalls}")
                out.append("")

        # ★ 解题步骤放在结果**之前**:这是给人看的部分,
        #   比原始的尝试记录重要得多。
        steps = list(getattr(self.solution, "steps", []) or [])
        if not brief and steps:
            out.append(mark("解题步骤"))
            for index, step in enumerate(steps, start=1):
                out.append(f"   第 {index} 步 · {step}")
            out.append("")

        out.append(mark("结果与验证"))
        if self.ok:
            out.append(f"   ∫_{{{self.lower}}}^{{{self.upper}}} {_short(self.f, 50)} dx"
                       f" = {_short(self.result)}")
            out.append(f"   LaTeX: {self.result_latex}")
            out.append(f"   ✅ 已验证 · 证据链:{getattr(self.solution, 'proof', '')}")
            out.append(f"   采用策略:{getattr(self.solution, 'strategy', '')}")
            if getattr(self.solution, "conditions", None):
                out.append("   ⚠ 成立条件:")
                for condition in self.solution.conditions:
                    out.append(f"      · {condition}")
        elif self.diverges:
            out.append("   ⛔ 该积分发散(不是「没算出来」,是它真的不收敛)")
            reason = getattr(self.solution, "divergence_reason", "")
            if reason:
                out.append(f"   判据:{reason}")
        else:
            out.append("   ❌ 未能给出通过数值验证的结果")
            out.append("   定积分的验证靠高精度数值求积,验证不过就不会给答案。")
        out.append("")

        numeric = getattr(self.solution, "numeric", None)
        if not brief and numeric is not None:
            out.append(mark("数值层(定积分的独立真值来源)"))
            out.append(f"   结论:{numeric.describe()}")
            for note in numeric.notes[-4:]:
                out.append(f"   · {note[:110]}")
            out.append("")

        if not brief and self.is_parametric:
            out.append(mark("含参积分专项"))
            from .parametric import describe_derivative_checks
            if self.derivative_checks:
                for param, checks in self.derivative_checks:
                    out.append(f"   参数 {param}:")
                    out.append("   " + describe_derivative_checks(checks).replace("\n", "\n   "))
            else:
                out.append("   未找到可用于求导检查的参数")
            if self.uniform_note:
                out.append("   " + self.uniform_note.replace("\n", "\n   "))
            out.append("")

        if not brief:
            out.append(mark(f"尝试记录(共 {len(getattr(self.solution, 'attempts', []))} 次)"))
            for attempt in getattr(self.solution, "attempts", []):
                mark = "✅" if attempt.verify and attempt.verify.ok else "❌"
                out.append(f"   {mark} {attempt.strategy}")
                out.append(f"        → {_short(attempt.value, 46) if attempt.value is not None else '—'}"
                           f"   {attempt.note[:90]}")
            for message in getattr(self.solution, "messages", []):
                out.append(f"   · {message[:120]}")
            out.append("")
        return "\n".join(out)

    def as_dict(self) -> dict:
        value = self.result
        return {
            "input": self.input_text,
            "kind": "definite",
            "integrand": str(self.f),
            "variable": str(self.x),
            "lower": str(self.lower),
            "upper": str(self.upper),
            "features": {key: sorted(value)
                         for key, value in self.features.as_match_dict().items()},
            "top1_card": self.top1_card_id,
            "top_hits": [{"id": h.id, "score": round(h.score, 4)} for h in self.hits],
            "strategy": getattr(self.solution, "strategy", ""),
            "result": str(value) if value is not None else None,
            "result_latex": self.result_latex,
            "ok": self.ok,
            "diverges": self.diverges,
            "conditions": list(getattr(self.solution, "conditions", [])),
            "proof": getattr(self.solution, "proof", ""),
            "steps": list(getattr(self.solution, "steps", []) or []),
            "numeric": (self.solution.numeric.describe()
                        if getattr(self.solution, "numeric", None) else ""),
            "parametric": self.is_parametric,
            "n_attempts": len(getattr(self.solution, "attempts", [])),
        }


def _short(expr, limit: int = 78) -> str:
    text = str(expr)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def analyze_definite(text: str | None = None, expr: sp.Expr | None = None,
                     lower=None, upper=None, var: str | None = None,
                     k: int = 4, retriever: Retriever | None = None,
                     corpus: str | Path | None = None,
                     parsed: IntegralInput | None = None,
                     supplementary: bool | None = None) -> DefiniteReport:
    """对一道定积分跑完整条流水线。

    supplementary 控制含参题的附加分析(积分号下求导的数值检查、
    尾部一致性探查):
        None(默认) 只在**它可能带来新信息**时才做 —— 见下方注释;
        True       强制做(最多花 SUPPLEMENTARY_BUDGET 秒);
        False      完全不做。
    """
    from .definite import solve_definite
    from .parametric import derivative_checks_for, probe_uniform_convergence

    if parsed is None:
        if expr is None:
            if not text:
                raise ParseError("必须提供 text 或 expr")
            parsed = parse_integral(text, var)
            input_text = text
        else:
            f = expr
            free = sorted((s for s in f.free_symbols if s.is_real), key=lambda s: s.name)
            x = next((s for s in free if s.name == "x"), free[0] if free else sp.Symbol("x"))
            parsed = IntegralInput(f, x, sp.sympify(lower), sp.sympify(upper))
            input_text = parsed.describe()
    else:
        input_text = text or parsed.describe()

    if not parsed.is_definite:
        raise ParseError("analyze_definite 需要上下限都给出")

    f, x, lo, hi = parsed.integrand, parsed.variable, parsed.lower, parsed.upper
    active = retriever or get_definite_retriever(corpus)
    feats = extract_definite(f, x, lo, hi)
    hits = active.retrieve(feats, k=k)
    plan = make_plan(f, x, feats, hits)
    solution = solve_definite(f, x, lo, hi, plan_subs=plan.substitutions)

    # 含参积分:额外做"积分号下求导"的数值检查和尾部一致性探查。
    # 这两项是**增值功能**,不是结果的必要条件 —— 所以必须套超时。
    # 它们内部要做 sp.diff、对 ∂f/∂t 数值求积、还要扫几组参数值,
    # 其中任何一步都可能意外地慢;绝不能让附加分析把主结果拖住。
    #
    # 关键一:两段**合起来共用一个预算**,而不是各给 15 秒。
    # 由 timeout.py 的预算传播机制自动分配 —— 第一段用掉多少,
    # 第二段就只剩多少。这样"增值分析"的总开销有了硬上界。
    #
    # 关键二(auto):**只在它可能带来新信息时才做**。
    # 主结果已经通过独立数值验证、而且它依赖的每个参数都已经报告了
    # 成立条件时,再把"积分号下求导"数值核对一遍不会改变结论,
    # 却要白等最多 SUPPLEMENTARY_BUDGET 秒。
    # 实测 ∫_0^∞ e^{-t x²}dx 的答案是秒出的,时间全花在这段冗余分析上。
    # 结论弱(没解出来,或者参数的成立范围还是未知)时才真正需要它 ——
    # 那时它是在"补证据",不是在"重复已经确定的结论"。
    derivative_checks = []
    uniform_note = ""
    if "parameterized" in feats.misc:
        free_params = sorted(
            (s for s in f.free_symbols if s is not x and s.is_real),
            key=lambda s: s.name)
        weak = (not solution.ok) or (bool(free_params) and not solution.conditions)
        wanted = supplementary if supplementary is not None else weak

        if not wanted:
            solution.messages.append(
                "主结果已通过数值验证且参数的成立条件已给出,"
                "跳过积分号下求导/尾部一致性的附加分析"
                "(无新信息,不必白等;需要时可传 supplementary=True 强制)")
        else:
            def supplementary_run():
                checks = derivative_checks_for(f, x, lo, hi) or []
                tail = ""
                if hi == sp.oo or lo == -sp.oo:
                    if free_params:
                        tail = probe_uniform_convergence(
                            f, x, free_params[0], lo if lo != -sp.oo else 0) or ""
                return checks, tail

            status, payload = run_with_timeout(supplementary_run,
                                               SUPPLEMENTARY_BUDGET)
            if status == "ok" and payload:
                derivative_checks, uniform_note = payload
            else:
                solution.messages.append(
                    "积分号下求导/尾部一致性的附加分析超时或失败,已跳过"
                    "(不影响上面的结果)")

    return DefiniteReport(input_text=input_text, f=f, x=x, lower=lo, upper=hi,
                          features=feats, hits=hits, plan=plan, solution=solution,
                          derivative_checks=derivative_checks,
                          uniform_note=uniform_note)


# ---------------------------------------------------------------- 主入口
def analyze(text: str | None = None,
            expr: sp.Expr | None = None,
            var: str | None = None,
            k: int = 4,
            corpus: str | Path | None = None,
            retriever: Retriever | None = None):
    """对一道积分跑完整条流水线。

    自动分派:带上下限走定积分(DefiniteReport),否则走不定积分(Report)。
    要么给 text(LaTeX 或纯文本),要么给 expr(已经是 SymPy 表达式)。
    """
    if expr is None:
        if not text:
            raise ParseError("必须提供 text 或 expr")
        parsed = parse_integral(text, var)
        if parsed.is_definite:
            return analyze_definite(text=text, parsed=parsed, k=k,
                                    retriever=retriever, corpus=corpus)
        f, x, input_text = parsed.integrand, parsed.variable, text
    else:
        f = expr
        if var is None:
            free = sorted((s for s in f.free_symbols if s.is_real), key=lambda s: s.name)
            x = next((s for s in free if s.name == "x"), free[0] if free else sp.Symbol("x"))
        else:
            x = sp.Symbol(var, real=True)
        input_text = f"∫{f} d{x}"

    active = retriever or get_retriever(corpus)
    feats = extract(f, x)
    hits = active.retrieve(feats, k=k,
                           query_text=input_text if not _looks_like_formula(input_text) else None)
    plan = make_plan(f, x, feats, hits)
    solution = solve(f, x, plan.substitutions)

    return Report(input_text=input_text, f=f, x=x, features=feats,
                  hits=hits, plan=plan, solution=solution)


def _looks_like_formula(text: str) -> bool:
    """粗略判断输入是公式还是自然语言描述 —— 决定要不要开启文本检索通道。"""
    if "\\" in text:
        return True
    alnum = sum(ch.isalnum() for ch in text)
    chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return chinese <= 3 or alnum > 2 * chinese


def solve_latex(text: str, var: str | None = None) -> sp.Expr | None:
    """一行式调用:给 LaTeX,返回原函数或 None。"""
    return analyze(text, var=var).result


def find_method(description: str, k: int = 3,
                corpus: str | Path | None = None,
                retriever: Retriever | None = None) -> list[Hit]:
    """按自然语言描述检索方法卡片。

    用于"根号里面是平方差该怎么做"这类只给了语言、没给式子的问法。
    这条路只走中文文本通道,不涉及结构特征。
    """
    active = retriever or get_retriever(corpus)
    return active.retrieve(Features(), k=k, query_text=description, text_only=True)
