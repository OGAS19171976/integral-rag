"""含参积分 —— 费曼技巧,以及"积分号下求导"的数值验证。

## 这一节要面对的东西

"含参积分"在数学分析里主要是一套**理论**:连续性、可微性、一致收敛,
以及最核心的那条结论 ——

    d/dt ∫_a^b f(x,t)dx  ==  ∫_a^b ∂f/∂t dx

这套系统能做的、并且**能验证**的,是下面三件:

1. **算出含参积分的闭式,并报出成立条件。**
   ∫_0^∞ x^{s−1}e^{−x}dx = Γ(s) 只在 s>0 成立 —— 条件不是废话,是答案的一部分。
   SymPy 通常会把条件放在 Piecewise 里,我们把它提出来。

2. **数值验证"积分号下求导"是否成立。**
   左边用中心差分数值求导,右边对 ∂f/∂t 直接做数值求积,在若干个 t 上对比。
   **必须说清楚:这不能证明一致收敛。** 它只能在有限个 t 上给出"看起来合法"
   的证据,以及在某个 t 上直接**否证**(比如 ∫_0^∞ sin(tx)/x dx = π/2,
   右边 ∫_0^∞ cos(tx)dx 发散,数值一算就知道交换次序不合法)。
   数学上的证明仍然要靠 Weierstrass 判别法那一套。

3. **用费曼技巧(对参数求导)真的把积分算出来。**
   对于 SymPy 直接算不动的 ∫_0^∞ (e^{−ax} − e^{−bx})/x dx 这类,
   对参数求导之后积分立刻变简单。常数项由"在某个特殊参数值上积分为 0"
   或者数值求积定出,最后仍然要过数值验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .quadrature import close, evaluate

_TRIG = (sp.sin, sp.cos)


@dataclass
class FeynmanResult:
    param: sp.Symbol
    value: sp.Expr
    evidence: str
    constant_source: str = ""


@dataclass
class DerivativeCheck:
    t_value: float
    left: complex | None          # d/dt ∫f dx(中心差分)
    right: complex | None         # ∫ ∂f/∂t dx
    agree: bool
    note: str = ""


# ================================================================ 参数
def parameters_of(f: sp.Expr, x: sp.Symbol) -> list[sp.Symbol]:
    return sorted((s for s in f.free_symbols if s is not x and s.is_real),
                  key=lambda s: s.name)


# ================================================================ 费曼技巧
def _constant_from_zero(f: sp.Expr, p: sp.Symbol, x: sp.Symbol,
                        H: sp.Expr, other_params: list[sp.Symbol]) -> tuple[sp.Expr, str] | None:
    """常数项优先用"某个参数值上积分恒为 0"这条精确理由来定。

    ∫_0^∞ (e^{−ax} − e^{−bx})/x dx 里把 a 取成 b,被积函数恒为 0,
    于是 I(b)=0,常数就被精确定下来了 —— 这就是教材里的做法,
    比数值凑常数干净得多。
    """
    for candidate in other_params:
        try:
            if sp.simplify(f.subs(p, candidate)) == 0:
                constant = sp.simplify(-H.subs(p, candidate))
                return constant, f"由 {p}={candidate} 时被积函数恒为 0(I=0)精确定出常数"
        except Exception:
            continue
    return None


def feynman(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr,
            max_params: int = 3) -> list[FeynmanResult]:
    """对每个参数尝试费曼技巧,返回可能的闭式(还没验证,由调用方验证)。"""
    params = parameters_of(f, x)
    if not params:
        return []

    results: list[FeynmanResult] = []
    for p in params[:max_params]:
        # ① 对参数求导
        try:
            g = sp.diff(f, p)
        except Exception:
            continue
        if sp.simplify(g) == 0:
            continue

        # ② 把 x 积掉(这一步通常变得很简单)
        try:
            G = sp.integrate(g, (x, lo, hi))
        except Exception:
            continue
        if G.has(sp.Integral) or G.has(sp.AccumBounds) or G.has(x):
            continue

        # ③ 再对参数积分回来
        try:
            H = sp.integrate(G, p)
        except Exception:
            continue
        if H.has(sp.Integral):
            continue

        others = [q for q in params if q is not p]

        # ④ 定常数
        exact = _constant_from_zero(f, p, x, H, others)
        if exact is not None:
            constant, source = exact
            value = sp.simplify(H + constant)
            results.append(FeynmanResult(
                p, value,
                f"∂f/∂{p} = {g} → ∫dx = {G} → 积分回 {p} 得 {H};{source}",
                source))
            continue

        # 数值定常数:取一个具体的参数值,先把 I 数值算出来
        probe = sp.Integer(1) if p.is_positive else sp.Float("1.5", 25)
        try:
            numeric = evaluate(f.subs(p, probe), x, lo, hi)
        except Exception:
            continue
        if not numeric.ok:
            continue
        try:
            constant = sp.simplify(sp.N(complex(numeric.value), 25) - H.subs(p, probe))
        except Exception:
            continue
        value = sp.simplify(H + constant)
        results.append(FeynmanResult(
            p, value,
            f"∂f/∂{p} = {g} → ∫dx = {G} → 积分回 {p} 得 {H};"
            f"常数由 I({p}={probe}) 的数值求积定出",
            f"数值定常数(p={probe})"))
    return results


# ================================================================ 积分号下求导的验证
def _numeric_integral(f: sp.Expr, x: sp.Symbol, lo, hi, t: sp.Symbol, t_value) -> complex | None:
    result = evaluate(f, x, lo, hi, params={t: t_value})
    if not result.ok:
        return None
    return complex(result.value)


def verify_derivative_under_integral(f: sp.Expr, x: sp.Symbol, t: sp.Symbol,
                                     lo: sp.Expr, hi: sp.Expr,
                                     t_values=None, rel_tol: float = 1e-6) -> list[DerivativeCheck]:
    """数值检查 d/dt ∫f dx 是否等于 ∫ ∂f/∂t dx。

    左边:中心差分数值求导。
    右边:对 ∂f/∂t 直接数值求积。

    返回每个采样 t 上的对比。任何一条 `agree=False` 都意味着
    **在这个 t 上交换次序不合法** —— 这是否证,是硬结论。
    全部 `agree=True` 只是"在这几个点上没发现问题",不是证明。
    """
    if t_values is None:
        t_values = [0.7, 1.3, 2.1]
    try:
        dfdt = sp.diff(f, t)
    except Exception:
        return []
    if sp.simplify(dfdt) == 0:
        return []

    checks: list[DerivativeCheck] = []
    for t0 in t_values:
        step = max(1e-7, abs(float(t0)) * 1e-7)
        plus = _numeric_integral(f, x, lo, hi, t, float(t0) + step)
        minus = _numeric_integral(f, x, lo, hi, t, float(t0) - step)
        right = _numeric_integral(dfdt, x, lo, hi, t, float(t0))

        if plus is None or minus is None:
            checks.append(DerivativeCheck(t0, None, right, False,
                                          "左边的数值积分算不出来,无法比较"))
            continue
        left = (plus - minus) / (2 * step)
        if right is None:
            checks.append(DerivativeCheck(t0, left, None, False,
                                          "右边 ∫ ∂f/∂t dx 数值上不存在(发散或无定义),"
                                          "说明在这个 t 上交换次序不合法"))
            continue
        agree, rel = close(left, right, rel_tol=rel_tol)
        checks.append(DerivativeCheck(
            t0, left, right, agree,
            f"左边 {left:.12g} vs 右边 {right:.12g}(相对误差 {rel:.2e})"))
    return checks


def describe_derivative_checks(checks: list[DerivativeCheck]) -> str:
    if not checks:
        return "无法进行积分号下求导的数值检查"
    bad = [c for c in checks if not c.agree]
    if bad:
        lines = [f"⚠ 积分号下求导在 {len(bad)}/{len(checks)} 个采样点上**不成立**(这是否证):"]
        for check in bad:
            lines.append(f"    t={check.t_value}:{check.note}")
        lines.append("    结论:不能把求导移到积分号里面。")
        return "\n".join(lines)
    lines = [f"○ 积分号下求导在 {len(checks)} 个采样点上数值一致:"
             f"d/dt ∫f dx ≈ ∫ ∂f/∂t dx"]
    for check in checks:
        lines.append(f"    t={check.t_value}:{check.note}")
    lines.append("    注意:这只是有限个点上的数值证据,**不构成一致收敛的证明**;"
                 "严格的证明仍需 Weierstrass 判别法一类工具。")
    return "\n".join(lines)


# ================================================================ 一致收敛的数值探查
def probe_uniform_convergence(f: sp.Expr, x: sp.Symbol, t: sp.Symbol,
                              lo: sp.Expr, tail_points=None) -> str:
    """对 [lo, ∞) 上的含参积分,数值探查"尾巴是否一致地小"。

    做法:取几个 A,估计 sup_t |∫_A^∞ f(x,t)dx|。如果随着 t 变化这个上界
    不肯趋于 0,就**否证**一致收敛。
    再次强调:这只能否证,不能证明。
    """
    if tail_points is None:
        tail_points = [5, 20, 80]
    t_values = [0.2, 1.0, 5.0]
    lines = ["○ 尾部一致性的数值探查(只能否证,不能证明):"]
    for A in tail_points:
        worst = None
        for t0 in t_values:
            value = _numeric_integral(f, x, A, sp.oo, t, t0)
            if value is None:
                worst = None
                break
            worst = abs(value) if worst is None else max(worst, abs(value))
        if worst is None:
            lines.append(f"    A={A}:尾巴算不出来(可能本身发散)")
        else:
            lines.append(f"    A={A}:sup_t |∫_A^∞ f dx| ≈ {worst:.3e}")
    return "\n".join(lines)


# ================================================================ 无穷限上的导数检查
def derivative_checks_for(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr):
    """给定一个含参积分,自动挑参数做求导合法性检查。"""
    params = parameters_of(f, x)
    checks = []
    for t in params:
        result = verify_derivative_under_integral(f, x, t, lo, hi)
        if result:
            checks.append((t, result))
    return checks


_ = field, _TRIG  # 保留导入的可读性
