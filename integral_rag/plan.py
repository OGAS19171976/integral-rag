"""规划层 —— 连接「检索」和「执行」。

检索层说:"这题大概是三角替换 x = a·sin t"。
规划层要做的是把这句话变成一个**能送进 CAS 的表达式**:x = a·sin(t),
让执行层去算 ∫f(a·sin t)·a·cos t dt,再换回来。

这里有一件事必须做对,否则整条代换路线会悄悄给出错误结果:

    **回代要给出"逆向映射",而不是只有 x = g(t)。**
    CAS 解反函数时经常给出不可读甚至带复数分支的结果
    (∫√(x²−a²)/x dx 的 asec(x/a) 会被 SymPy 化成
     I*(log(...) − tanh(log(...))) 这种怪物)。而三角替换的逆向关系
    是确定的:sin t = x/a、cos t = √(a²−x²)/a …… 直接写出来,
    得到的就是教材里的标准形式。

另外欧拉替换需要 pre_subs:sqrt(ax²+bx+c) 不是"把 x 换掉就能化简"的,
必须先显式把根式换成 t − √a·x。

不是每张卡片都能变成代换。分部积分、递推公式、配对技巧这些是"代数操作"
而不是"变量替换",它们的价值在于给人和给大模型看的指导文本;
真正落到执行上的部分,由执行层的恒等变形策略覆盖。这一点在报告里会如实标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .features import Features
from .solve import Substitution

_FRESH_NAMES = ("t", "u", "v", "w", "s")


def _fresh_var(f: sp.Expr) -> sp.Symbol:
    for name in _FRESH_NAMES:
        sym = sp.Symbol(name, real=True)
        if sym not in f.free_symbols:
            return sym
    return sp.Symbol("_t0", real=True)


# ---------------------------------------------------------------- 根号信息
def _radical_roots(f: sp.Expr, x: sp.Symbol) -> list[tuple[str, object, sp.Expr]]:
    """找出所有二次/一次根式,返回 [(类型, 参数, 根号内的底数)]。

    a2-x2 / x2+a2 / x2-a2 → 参数是 a(根号内常数项的平方根)
    ax+b                  → 参数是 (a, b)
    quadratic             → 参数是 (a, b, c)
    """
    out: list[tuple[str, object, sp.Expr]] = []
    for node in sp.preorder_traversal(f):
        if not (isinstance(node, sp.Pow) and node.exp.is_Rational and node.exp.q == 2):
            continue
        base = node.base
        try:
            poly = sp.Poly(base, x)
        except Exception:
            continue
        deg = poly.degree()
        if deg == 1:
            a = poly.coeff_monomial(x)
            b = poly.coeff_monomial(1)
            out.append(("ax+b", (a, b), base))
        elif deg == 2:
            c2 = poly.coeff_monomial(x**2)
            c1 = poly.coeff_monomial(x)
            c0 = poly.coeff_monomial(1)
            if c1 != 0:
                out.append(("quadratic", (c2, c1, c0), base))
            elif c2.is_negative and c0.is_positive:
                out.append(("a2-x2", sp.sqrt(c0), base))
            elif c2.is_positive and c0.is_positive:
                out.append(("x2+a2", sp.sqrt(c0), base))
            elif c2.is_positive and c0.is_negative:
                out.append(("x2-a2", sp.sqrt(-c0), base))
            else:
                out.append(("quadratic", (c2, c1, c0), base))
    return out


def _radical_power_subs(f: sp.Expr, x: sp.Symbol, base: sp.Expr,
                        replacement: sp.Expr) -> list[tuple[sp.Expr, sp.Expr]]:
    """把 f 里所有 base 的半整数次幂 base^(k/2) 换成 replacement^k。

    因为 sqrt(base) = replacement,所以 base^(k/2) = replacement^k。

    为什么要按"幂"而不是按"根号节点"来找?因为被积函数里出现的很可能是
    base^(-1/2)(也就是 1/√base),这时要换成 1/replacement。
    之前只找"正次幂的 sqrt 节点",结果在 1/(x√(x²+x+1)) 里找到的
    其实是它自己的倒数,替换进去整个式子就错了 —— 而且错得很隐蔽:
    照样能积出一个表达式,只是原函数不对。
    """
    found: dict[str, tuple[sp.Expr, sp.Expr]] = {}
    for node in sp.preorder_traversal(f):
        if not isinstance(node, sp.Pow):
            continue
        exponent = node.exp
        if not exponent.is_Rational or exponent.q != 2:
            continue
        try:
            if sp.simplify(node.base - base) != 0:
                continue
        except Exception:
            continue
        found[str(node)] = (node, replacement ** (2 * exponent))
    return list(found.values())


# ---------------------------------------------------------------- 代换生成器
def _gen_trig_sin(f, x, feats):
    """x = a·sin t,处理 √(a²−x²)。"""
    for kind, a, _base in _radical_roots(f, x):
        if kind != "a2-x2":
            continue
        t = _fresh_var(f)
        root = sp.sqrt(a**2 - x**2)
        back = {
            sp.sin(t): x / a,
            sp.cos(t): root / a,
            sp.tan(t): x / root,
        }
        return Substitution(f"三角替换 x = {a}·sin t", a * sp.sin(t), t,
                            "trig_sub_a2_minus_x2",
                            "√(a²−x²) 用 sin 消去根号",
                            back_map=back, t_inverse=sp.asin(x / a))
    return None


def _gen_trig_tan(f, x, feats):
    """x = a·tan t,处理 √(x²+a²)。"""
    for kind, a, _base in _radical_roots(f, x):
        if kind != "x2+a2":
            continue
        t = _fresh_var(f)
        root = sp.sqrt(x**2 + a**2)
        back = {
            sp.tan(t): x / a,
            sp.sec(t): root / a,
            sp.sin(t): x / root,
            sp.cos(t): a / root,
        }
        return Substitution(f"三角替换 x = {a}·tan t", a * sp.tan(t), t,
                            "trig_sub_x2_plus_a2",
                            "√(x²+a²) 用 tan 消去根号",
                            back_map=back, t_inverse=sp.atan(x / a))
    return None


def _gen_trig_sec(f, x, feats):
    """x = a·sec t,处理 √(x²−a²)。"""
    for kind, a, _base in _radical_roots(f, x):
        if kind != "x2-a2":
            continue
        t = _fresh_var(f)
        root = sp.sqrt(x**2 - a**2)
        back = {
            sp.sec(t): x / a,
            sp.tan(t): root / a,
            sp.sin(t): root / x,
            sp.cos(t): a / x,
        }
        return Substitution(f"三角替换 x = {a}·sec t", a * sp.sec(t), t,
                            "trig_sub_x2_minus_a2",
                            "√(x²−a²) 用 sec 消去根号",
                            back_map=back, t_inverse=sp.asec(x / a))
    return None


def _gen_hyperbolic(f, x, feats):
    """x = a·sinh t,处理 √(x²+a²) 的另一条路线。"""
    for kind, a, _base in _radical_roots(f, x):
        if kind != "x2+a2":
            continue
        t = _fresh_var(f)
        root = sp.sqrt(x**2 + a**2)
        back = {
            sp.sinh(t): root / a,
            sp.cosh(t): x / a,
            sp.tanh(t): x / root,
        }
        return Substitution(f"双曲替换 x = {a}·sinh t", a * sp.sinh(t), t,
                            "hyperbolic_substitution",
                            "√(x²+a²) 用 sinh,回代用 arsinh",
                            back_map=back, t_inverse=sp.asinh(x / a))
    return None


def _gen_radical_linear(f, x, feats):
    """t = √(ax+b),处理一次根式。"""
    for kind, params, _base in _radical_roots(f, x):
        if kind != "ax+b":
            continue
        a, b = params
        if sp.simplify(a) == 0:
            continue
        t = _fresh_var(f)
        return Substitution(f"根式代换 t = √({a}x+{b})", (t**2 - b) / a, t,
                            "radical_linear_substitution",
                            "把一次根式整体设为 t,根号消失",
                            t_inverse=sp.sqrt(a * x + b))
    return None


def _gen_euler_first(f, x, feats):
    """欧拉第一替换:t = √a·x + √(ax²+bx+c)。

    关键在于它**可以显式解出 x**:
        t − √a·x = √(ax²+bx+c)
        t² − 2√a·t·x + a·x² = ax² + bx + c
        x = (t² − c) / (2√a·t + b)
    于是二次根式被彻底有理化(√(ax²+bx+c) = t − √a·x),积分变成纯有理函数,
    不引入 |·|、不引入 sec/tan 的复合。实测 ∫dx/(x√(x²+x+1)) 走"配方后三角替换"
    积不出来,走这条路一步就化成 ∫2/(t²−1)dt。
    """
    for kind, params, base in _radical_roots(f, x):
        if kind != "quadratic":
            continue
        a, b, c = params
        if not a.is_positive:
            continue
        t = _fresh_var(f)
        root_a = sp.sqrt(a)
        replacement = t - root_a * x
        pre = _radical_power_subs(f, x, base, replacement)
        if not pre:
            continue
        x_expr = (t**2 - c) / (2 * root_a * t + b)
        return Substitution("欧拉第一替换 t = √a·x + √(ax²+bx+c)", x_expr, t,
                            "euler_substitution",
                            "欧拉第一替换把二次根式整体有理化",
                            pre_subs=pre,
                            t_inverse=x + sp.sqrt(a * x**2 + b * x + c))
    return None


def _gen_quadratic_denominator_tan(f, x, feats):
    """分母含 1+x²、区间是 [0,1] 时,令 x = tan t。

    这是 ∫_0^1 ln(1+x)/(1+x²)dx 这类题的关键一步:换元之后变成
    ∫_0^{π/4} ln(1+tan t)dt,再用区间再现公式就能积出来。

    注意真正起作用的是**换元 + 区间再现的组合** —— 单靠换元还积不动,
    所以执行层在换元之后会把整套策略在新变量上**递归**再跑一遍。
    区间信息从 feats.interval 拿([0,1] 会被标成 "unit")。
    """
    from .features import _poly_degree_in

    if "unit" not in feats.interval:
        return None
    try:
        denominator = sp.denom(sp.together(f))
        factors = sp.factor(denominator).as_powers_dict()
    except Exception:
        return None

    for factor in factors:
        if _poly_degree_in(factor, x) != 2:
            continue
        try:
            poly = sp.Poly(factor, x)
        except Exception:
            continue
        if poly.coeff_monomial(x) != 0:
            continue
        c2 = poly.coeff_monomial(x**2)
        c0 = poly.coeff_monomial(1)
        if not (c2.is_positive and c0.is_positive):
            continue
        a = sp.sqrt(sp.simplify(c0 / c2))
        if sp.simplify(a - 1) != 0:      # 只有 1+x² 这种 a=1 的才对上 [0,1]
            continue
        t = _fresh_var(f)
        root = sp.sqrt(x**2 + 1)
        back = {sp.tan(t): x, sp.sec(t): root,
                sp.sin(t): x / root, sp.cos(t): 1 / root}
        return Substitution("三角替换 x = tan t(分母含 1+x²)", sp.tan(t), t,
                            "def_substitution_limits",
                            "把 1+x² 换成 sec²t,常与区间再现公式配合",
                            back_map=back, t_inverse=sp.atan(x))
    return None


def _gen_completing_square(f, x, feats):
    """√(ax²+bx+c) 配方后做三角替换(欧拉替换之外的备选路线)。"""
    for kind, params, _base in _radical_roots(f, x):
        if kind != "quadratic":
            continue
        a, b, c = params
        try:
            disc = sp.simplify(b**2 - 4 * a * c)
        except Exception:
            continue
        if not disc.is_negative or not a.is_positive:
            continue
        t = _fresh_var(f)
        k = sp.sqrt(-disc) / (2 * a)
        shift = b / (2 * a)
        return Substitution(f"配方后三角替换 x = {sp.simplify(k)}·tan t − {sp.simplify(shift)}",
                            k * sp.tan(t) - shift, t, "euler_substitution",
                            "配方成 k²tan²t + h² 形式,根号变成 sec 的有理式",
                            t_inverse=sp.atan((x + shift) / k))
    return None


def _gen_weierstrass(f, x, feats):
    """万能代换 t = tan(x/2)。"""
    if "rational_in_sin_cos" not in feats.forms:
        return None
    t = _fresh_var(f)
    return Substitution("万能代换 t = tan(x/2)", 2 * sp.atan(t), t,
                        "weierstrass_substitution",
                        "三角有理式统一化为 t 的有理函数",
                        t_inverse=sp.tan(x / 2))


def _gen_exp_rational(f, x, feats):
    """t = eˣ。"""
    if "exponential_rational" not in feats.misc:
        return None
    t = _fresh_var(f)
    return Substitution("指数代换 t = eˣ", sp.log(t), t,
                        "exp_rational_substitution",
                        "把 eˣ 设为 t,化为有理函数积分",
                        t_inverse=sp.exp(x))


def _gen_reciprocal(f, x, feats):
    """倒代换 x = 1/t。"""
    if feats.poly_degree is None:
        return None
    t = _fresh_var(f)
    return Substitution("倒代换 x = 1/t", 1 / t, t,
                        "reciprocal_substitution",
                        "分母次数明显高于分子时,倒代换常能化简",
                        t_inverse=1 / x)


def _gen_sin_cos_odd(f, x, feats):
    """sinᵐcosⁿ 中有一个奇次:把奇次的那个凑成微分。"""
    if "odd_power_sin_cos" not in feats.misc:
        return None
    m = n = 0
    for factor in f.as_ordered_factors():
        if factor == sp.sin(x):
            m += 1
        elif factor == sp.cos(x):
            n += 1
        elif isinstance(factor, sp.Pow) and factor.args[1].is_Integer:
            base, power = factor.args
            if base == sp.sin(x):
                m += int(power)
            elif base == sp.cos(x):
                n += int(power)
    t = _fresh_var(f)
    if n % 2 == 1:
        return Substitution("凑微分 u = sin x", sp.asin(t), t,
                            "sin_cos_odd_power",
                            "cos 的幂次为奇数,把 cos x dx 凑进微分",
                            t_inverse=sp.sin(x))
    if m % 2 == 1:
        return Substitution("凑微分 u = cos x", sp.acos(t), t,
                            "sin_cos_odd_power",
                            "sin 的幂次为奇数,把 sin x dx 凑进微分",
                            t_inverse=sp.cos(x))
    return None


def _gen_tan_sec(f, x, feats):
    """tan/sec 幂次型:t = tan x。"""
    if "trig" not in feats.families:
        return None
    if not f.has(sp.tan, sp.sec):
        return None
    t = _fresh_var(f)
    return Substitution("三角代换 t = tan x", sp.atan(t), t,
                        "tan_sec_powers", "tan/sec 幂次型化为 t 的有理函数",
                        t_inverse=sp.tan(x))


# 卡片 id -> 生成器。没有登记在这张表里的卡片,只贡献方法指导文本。
_GENERATORS = {
    "trig_sub_a2_minus_x2": _gen_trig_sin,
    "trig_sub_x2_plus_a2": _gen_trig_tan,
    "trig_sub_x2_minus_a2": _gen_trig_sec,
    "hyperbolic_substitution": _gen_hyperbolic,
    "radical_linear_substitution": _gen_radical_linear,
    "weierstrass_substitution": _gen_weierstrass,
    "denom_a_plus_b_trig": _gen_weierstrass,
    "exp_rational_substitution": _gen_exp_rational,
    "reciprocal_substitution": _gen_reciprocal,
    "sin_cos_odd_power": _gen_sin_cos_odd,
    "tan_sec_powers": _gen_tan_sec,
    "euler_substitution": _gen_euler_first,
    "algebraic_rationalization": _gen_radical_linear,
    "chebyshev_binomial": _gen_radical_linear,
    # ---- 定积分卡片
    "def_substitution_limits": _gen_quadratic_denominator_tan,
}

# 兜底生成器:即使检索没把它们排进前几名,只要特征对得上就试一下。
# 顺序有讲究 —— 欧拉第一替换(纯有理化)通常比拼方程的三角替换好走。
_FALLBACKS = (_gen_euler_first, _gen_completing_square, _gen_quadratic_denominator_tan,
              _gen_weierstrass, _gen_exp_rational)


@dataclass
class Plan:
    substitutions: list[Substitution] = field(default_factory=list)
    guidance: list[dict] = field(default_factory=list)   # 检索到的方法卡片
    actionable: list[str] = field(default_factory=list)  # 变成了代换的卡片 id
    advisory: list[str] = field(default_factory=list)    # 只提供指导的卡片 id
    notes: list[str] = field(default_factory=list)


def make_plan(f: sp.Expr, x: sp.Symbol, feats: Features, hits, max_subs: int = 4) -> Plan:
    plan = Plan(guidance=[h.card for h in hits])

    seen: set[str] = set()

    def add(sub: Substitution | None) -> bool:
        if sub is None:
            return False
        key = str(sub.x_expr)
        if key in seen:
            return False
        seen.add(key)
        plan.substitutions.append(sub)
        if sub.card_id and sub.card_id not in plan.actionable:
            plan.actionable.append(sub.card_id)
        return True

    for hit in hits:
        if len(plan.substitutions) >= max_subs:
            break
        generator = _GENERATORS.get(hit.id)
        if generator is None:
            if hit.id not in plan.advisory:
                plan.advisory.append(hit.id)
            continue
        try:
            if not add(generator(f, x, feats)):
                if hit.id not in plan.advisory:
                    plan.advisory.append(hit.id)
        except Exception as exc:
            plan.notes.append(f"生成代换时出错({hit.id}):{type(exc).__name__}")

    # 兜底:检索没覆盖到,但特征明确指向某个代换的情况
    for generator in _FALLBACKS:
        if len(plan.substitutions) >= max_subs:
            break
        try:
            sub = generator(f, x, feats)
        except Exception:
            continue
        if add(sub):
            plan.notes.append(f"兜底补充代换:{sub.name}")

    if not plan.substitutions:
        plan.notes.append("没有可执行的代换候选,将只依赖 CAS 内置算法与恒等变形策略")
    return plan


def describe_plan(plan: Plan) -> str:
    lines = []
    if plan.substitutions:
        lines.append("  执行层将依次尝试的代换:")
        for sub in plan.substitutions:
            tag = f"  ← 卡片 {sub.card_id}" if sub.card_id else "  ← 兜底"
            lines.append(f"    · {sub.name}{tag}")
    if plan.advisory:
        lines.append(f"  只提供方法指导(非代换型)的卡片:{', '.join(plan.advisory)}")
    for note in plan.notes:
        lines.append(f"  注:{note}")
    return "\n".join(lines)
