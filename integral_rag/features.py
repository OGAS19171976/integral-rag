"""被积函数的「结构特征」提取 —— 积分检索真正的信号源。

为什么不能靠文本相似度?
    "x*exp(x)" 和 "x*log(x)" 的字符重合度很高,但一个用分部积分
    (把 exp 塞进微分),另一个也用分部但方向相反(把 log 留下、多项式塞进去),
    而 "x*sqrt(1-x^2)" 字符上跟它们毫无关系,方法却是完全不同的三角替换。
    LaTeX 字符串的相似度对"该用哪个积分方法"几乎没有预测力。

真正有预测力的是**结构**:根号里是 a²-x² 还是 x²+a²、外层是不是有理函数、
有没有多项式因子乘指数、sin/cos 是不是有理式……这些决定了方法。
所以这里把被积函数映射成一个离散的"结构签名",检索时按签名匹配方法卡片。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import sympy as sp

TRIG_FUNCS = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
INV_TRIG_FUNCS = (sp.asin, sp.acos, sp.atan, sp.acot)
HYPER_FUNCS = (sp.sinh, sp.cosh, sp.tanh, sp.coth)

# has_radical 的取值域(和语料卡片里的字符串必须一致)
RADICAL_KINDS = ("a2-x2", "x2+a2", "x2-a2", "ax+b", "general")


# ---------------------------------------------------------------- 数据结构
@dataclass
class Features:
    """一个被积函数的结构签名。

    不定积分只用到前五个字段;定积分还会填 `interval`(区间类型),
    因为定积分的方法很大程度上由**区间**决定:
    对称区间想到奇偶性、[0, π/2] 想到 Wallis/Beta、[0, ∞) 想到收敛性。
    """

    families: set[str] = field(default_factory=set)   # 参与的函数族
    forms: set[str] = field(default_factory=set)      # 外层形态
    radicals: set[str] = field(default_factory=set)   # 根号类型
    misc: set[str] = field(default_factory=set)       # 组合信号
    interval: set[str] = field(default_factory=set)   # 区间类型(只有定积分才有)
    poly_degree: int | None = None
    tokens: Counter = field(default_factory=Counter)  # 节点直方图,用于结构相似度

    def as_match_dict(self) -> dict[str, set[str]]:
        """转成和卡片 triggers 对齐的字段,检索层直接用。"""
        return {
            "interval": self.interval,
            "has_radical": self.radicals,
            "has_func": self.families,
            "form": self.forms,
            "misc": self.misc,
        }

    def describe(self) -> str:
        """给终端看的中文摘要。"""
        label = {
            "poly": "多项式", "rational": "有理函数", "exp": "指数",
            "log": "对数", "trig": "三角", "inv_trig": "反三角",
            "radical": "根式", "hyperbolic": "双曲", "const": "常数",
            "abs": "绝对值",
        }
        form_label = {
            "sum": "和式", "product": "乘积", "quotient": "商式", "power": "幂",
            "composite": "复合", "rational": "有理式",
            "rational_in_sin_cos": "三角有理式", "any": "任意",
        }
        radical_label = {
            "a2-x2": "√(a²−x²)型", "x2+a2": "√(x²+a²)型",
            "x2-a2": "√(x²−a²)型", "ax+b": "√(ax+b)型", "general": "一般二次根式",
        }
        interval_label = {
            "symmetric": "关于原点对称", "zero_to_inf": "[0,∞)",
            "zero_to_pi_over_2": "[0,π/2]", "zero_to_pi": "[0,π]",
            "zero_to_two_pi": "[0,2π]", "unit": "[0,1]", "infinite": "无穷区间",
            "finite": "有限区间", "any": "任意",
        }
        parts = []
        if self.interval:
            parts.append("区间:" + "/".join(interval_label.get(i, i) for i in sorted(self.interval)))
        if self.families:
            parts.append("函数族:" + "/".join(label.get(f, f) for f in sorted(self.families)))
        if self.forms:
            parts.append("形态:" + "/".join(form_label.get(f, f) for f in sorted(self.forms)))
        if self.radicals:
            parts.append("根号:" + "/".join(radical_label.get(r, r) for r in sorted(self.radicals)))
        if self.misc:
            parts.append("信号:" + "/".join(sorted(self.misc)))
        if self.poly_degree:
            parts.append(f"多项式次数:{self.poly_degree}")
        return ";".join(parts) if parts else "无显著特征"


# ---------------------------------------------------------------- 小工具
def _is_sqrt(node) -> bool:
    return isinstance(node, sp.Pow) and node.exp.is_Rational and node.exp.q == 2


def _func_atoms(expr, funcs) -> list:
    out = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, funcs):
            out.append(node)
    return out


def _poly_degree_in(expr, x) -> int | None:
    """expr 作为 x 的多项式的次数;不是多项式返回 None。"""
    try:
        if not expr.is_polynomial(x):
            return None
        deg = sp.degree(sp.Poly(expr, x))
        return int(deg)
    except Exception:
        return None


def _top_level_args(expr):
    """把 Mul / Add 的因子(或项)摊平;其他情况返回 [expr]。"""
    if isinstance(expr, sp.Mul):
        return list(expr.args)
    if isinstance(expr, sp.Add):
        return list(expr.args)
    return [expr]


def _contains_func(expr, funcs) -> bool:
    try:
        return bool(expr.has(*funcs))
    except Exception:
        return False


# ---------------------------------------------------------------- 根号分类
def _classify_radical(base, x) -> str:
    """判断根号内部属于哪一类:a²-x² / x²+a² / x²-a² / ax+b / general。"""
    deg = _poly_degree_in(base, x)
    if deg is None:
        return "general"
    try:
        poly = sp.Poly(base, x)
    except Exception:
        return "general"

    if deg == 1:
        return "ax+b"
    if deg != 2:
        return "general"

    try:
        c2 = poly.coeff_monomial(x**2)
        c1 = poly.coeff_monomial(x)
        c0 = poly.coeff_monomial(1)
    except Exception:
        return "general"

    if sp.simplify(c1) != 0:          # 有一次项 → 配方后属于一般二次式
        return "general"

    c2_pos = bool(c2.is_positive)
    c0_pos = bool(c0.is_positive)
    c0_neg = bool(c0.is_negative)

    if not c2_pos:                     # −x² + a²  →  a²−x²
        return "a2-x2" if c0_pos else "general"
    if c0_pos:
        return "x2+a2"
    if c0_neg:
        return "x2-a2"
    return "general"


# ---------------------------------------------------------------- 有理三角式
def _denominator_uses(swapped, dummies) -> bool:
    """化简后的分母里,是否真的出现了那些哑符号。

    这是"是不是有理式"这句话的关键限定。以 ∫cos x dx 为例:
    把 cos x 换成哑符号 d 之后得到的就是 d 本身 —— 它的分母是 1,
    也就是说 cos x 只是 cos 的**多项式**,谈不上"cos 的有理式",
    真正该用的方法是基本积分表,而不是万能代换。
    只有分母里出现了 sin/cos(t = 1/(2+cos x) 这类)时,
    "三角有理式"这个判断才真的有意义。
    """
    try:
        denominator = sp.denom(sp.cancel(swapped))
    except Exception:
        return True
    return any(dummy in denominator.free_symbols for dummy in dummies)


def _is_rational_in_trig(expr, x) -> bool:
    """被积函数是不是「sin(x)/cos(x) 的有理式」,如 1/(2+cos x)。

    做法:把每个 sin(x)/cos(x)/tan(x)… 整体换成一个哑符号,
    如果换完之后 x 不再出现、且关于哑符号是有理函数(分母真的用到哑符号),
    就成立。
    """
    atoms = _func_atoms(expr, TRIG_FUNCS)
    if not atoms:
        return False
    # 只处理自变量恰好是 x 的情形(教材里的标准形式)
    atoms = [a for a in atoms if a.args == (x,)]
    if not atoms:
        return False

    subs = {a: sp.Symbol(f"_d{i}", real=True) for i, a in enumerate(sorted(set(atoms), key=str))}
    swapped = expr.subs(subs)
    dummies = list(subs.values())
    if x in swapped.free_symbols:
        return False
    try:
        if not swapped.is_rational_function(*dummies):
            return False
    except Exception:
        return False
    return _denominator_uses(swapped, dummies)


def _is_rational_in_exp(expr, x) -> bool:
    """是不是 eˣ 的有理式,如 1/(1+e^x)、e^x/(e^x+1)。

    同样要求分母真的用到 eˣ:∫eˣdx、∫e^{2x−1}dx 都属于"基本表/凑微分",
    不是"eˣ 的有理式",不该被推荐去令 t = eˣ。
    """
    atoms = [n for n in sp.preorder_traversal(expr) if isinstance(n, sp.exp)
             or (isinstance(n, sp.Pow) and n.base is sp.E)]
    atoms = [a for a in atoms if a.has(x)]
    if not atoms:
        return False
    subs = {a: sp.Symbol(f"_e{i}", real=True) for i, a in enumerate(sorted(set(atoms), key=str))}
    swapped = expr.subs(subs)
    dummies = list(subs.values())
    if x in swapped.free_symbols:
        return False
    try:
        if not swapped.is_rational_function(*dummies):
            return False
    except Exception:
        return False
    return _denominator_uses(swapped, dummies)


def _is_nontrivial_linear(expr, x) -> bool:
    """是不是"非恒等"的一次式 ax+b(即内层真的做了一次变换)。"""
    if _poly_degree_in(expr, x) != 1:
        return False
    try:
        poly = sp.Poly(expr, x)
        a = poly.coeff_monomial(x)
        b = poly.coeff_monomial(1)
    except Exception:
        return False
    return not (sp.simplify(a - 1) == 0 and sp.simplify(b) == 0)


def _has_linear_inner(expr, x) -> bool:
    """式子里是否存在 f(ax+b) 这样的复合(内层是非恒等一次式)。

    这是"凑微分/第一类换元"最本质的判别信号:
    ∫sin(3x+1)dx、∫dx/(2x+5)、∫(2x+1)⁵dx 都属于这一类,
    而 ∫sin x dx、∫dx/x 不是 —— 后者直接查基本表就行。
    只看外层形态(Add/Mul/Pow)区分不出这两种情况。
    """
    outers = TRIG_FUNCS + INV_TRIG_FUNCS + HYPER_FUNCS + (sp.exp, sp.log)
    for node in sp.preorder_traversal(expr):
        if isinstance(node, outers) and node.args:
            if _is_nontrivial_linear(node.args[0], x):
                return True
        elif isinstance(node, sp.Pow):
            # 1/(2x+5)、(2x+1)^5 都算
            if _is_nontrivial_linear(node.base, x):
                return True
    return False


def _sin_cos_power_info(expr, x):
    """若 expr 形如 sin(x)^m · cos(x)^n,返回 (m, n);否则 None。"""
    m = n = 0
    for factor in expr.as_ordered_factors():
        if factor == sp.sin(x):
            m += 1
            continue
        if factor == sp.cos(x):
            n += 1
            continue
        if isinstance(factor, sp.Pow) and factor.args[1].is_Integer:
            base, power = factor.args
            if base == sp.sin(x):
                m += int(power)
                continue
            if base == sp.cos(x):
                n += int(power)
                continue
        return None
    if m == 0 and n == 0:
        return None
    return m, n


def _exp_args(expr) -> list:
    """收集所有以 x 为自变量的指数部分的指数(同时覆盖 exp(u) 和 E**u)。"""
    out = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.exp):
            out.append(node.args[0])
        elif isinstance(node, sp.Pow) and node.base is sp.E:
            out.append(node.exp)
    return out


def _trig_args(expr, funcs=TRIG_FUNCS) -> list:
    return [n.args[0] for n in sp.preorder_traversal(expr)
            if isinstance(n, funcs) and len(n.args) == 1]


def _any_linear_arg(args, x) -> bool:
    for arg in args:
        if _poly_degree_in(arg, x) in (0, 1):
            return True
    return False


def _is_trig_product_to_sum(expr, x) -> bool:
    """sin(ax)·cos(bx)(a≠b)这类「乘积化和差」信号。

    必须要求**两个不同的角**。sin(x)·cos(x)、cos(x)/sin(x) 这类同角表达式
    虽然也含两个三角函数,但它们该用凑微分(或倍角公式)处理,
    化成和差反而绕远路 —— 之前没做这个限制,导致 ∫cos x/sin x dx
    被推荐去"积化和差"。
    """
    linear = []
    for f in expr.as_ordered_factors():
        if isinstance(f, TRIG_FUNCS) and len(f.args) == 1:
            arg = f.args[0]
            try:
                if sp.degree(arg, x) in (0, 1):
                    linear.append((type(f).__name__, sp.simplify(arg)))
            except Exception:
                continue
    if len(linear) < 2:
        return False
    args = {arg for _, arg in linear}
    return len(args) >= 2


def _binomial_power_match(expr, x):
    """匹配 x^m·(a+b·x^n)^p(切比雪夫二项式微分)。

    要求:除了一个分数次幂因子之外,其余因子都是关于 x 的**单项式**,
    而那个分数次幂的底数是关于 x 的**二项式**。
    例:x²·√(1+x³) → m=2, n=3, p=1/2。
    """
    if not isinstance(expr, sp.Mul):
        return None

    power_part = None
    monomial_degree = 0

    for factor in expr.as_ordered_factors():
        if isinstance(factor, sp.Pow) and factor.exp.is_Rational and factor.exp.q != 1:
            power_part = factor
            continue
        if _poly_degree_in(factor, x) is None:
            return None                      # 出现了非多项式因子,不属于该形式
        try:
            poly = sp.Poly(factor, x)
        except Exception:
            return None
        if len(poly.terms()) != 1:           # 不是单项式
            return None
        monomial_degree += int(poly.degree())

    if power_part is None:
        return None

    base, exponent = power_part.args
    try:
        base_poly = sp.Poly(base, x)
    except Exception:
        return None
    if len(base_poly.terms()) != 2 or base_poly.degree() < 1:
        return None

    return (monomial_degree, int(base_poly.degree()), exponent)


# ---------------------------------------------------------------- 主入口
def extract(expr: sp.Expr, x: sp.Symbol) -> Features:
    """提取结构签名。这是整条流水线的第 2 步。"""
    feats = Features()

    # ---- 节点直方图(结构相似度用)
    hist: Counter = Counter()
    for node in sp.preorder_traversal(expr):
        name = type(node).__name__
        hist[name] += 1
        if _is_sqrt(node):
            hist["___sqrt"] += 1
        if isinstance(node, sp.Pow) and node.exp.is_negative:
            hist["___negative_power"] += 1
    feats.tokens = hist

    # ---- 函数族
    degree = _poly_degree_in(expr, x)
    feats.poly_degree = degree
    is_rational_fn = False
    try:
        is_rational_fn = bool(expr.is_rational_function(x))
    except Exception:
        pass

    if degree is not None and degree >= 1:
        feats.families.add("poly")
    if is_rational_fn and degree is None:
        feats.families.add("rational")
        feats.forms.add("rational")
        # 分子里若有次数≥1 的多项式,也标上 poly
        num, _den = sp.fraction(sp.cancel(expr))
        nd = _poly_degree_in(num, x)
        if nd and nd >= 1:
            feats.families.add("poly")

    if _contains_func(expr, (sp.exp,)) or (expr.has(sp.E) and any(
            isinstance(n, sp.Pow) and n.base is sp.E for n in sp.preorder_traversal(expr))):
        feats.families.add("exp")
    if _contains_func(expr, (sp.log,)):
        feats.families.add("log")
    if _contains_func(expr, TRIG_FUNCS):
        feats.families.add("trig")
    if _contains_func(expr, INV_TRIG_FUNCS):
        feats.families.add("inv_trig")
    if _contains_func(expr, HYPER_FUNCS):
        feats.families.add("hyperbolic")
    if not expr.free_symbols - {x} and not expr.has(x):
        feats.families.add("const")

    # ---- 根式
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp.is_Rational and node.exp.q == 2:
            feats.radicals.add(_classify_radical(node.base, x))
    if feats.radicals:
        feats.families.add("radical")

    # ---- 外层形态
    # 常值被积函数(如 ∫5dx)没有"外层形态"可言。原来它落到 else 分支被标成
    # composite,于是会去匹配那些要求 form:composite 的卡片
    # (比如凑微分),而正确答案只是查一下基本积分表。
    if not expr.has(x):
        pass
    elif is_rational_fn and degree is None:
        pass                                  # 已经加过 rational
    elif isinstance(expr, sp.Add):
        feats.forms.add("sum")
    elif isinstance(expr, sp.Mul):
        feats.forms.add("product")
    elif isinstance(expr, sp.Pow):
        feats.forms.add("power")
    else:
        feats.forms.add("composite")
    if _has_linear_inner(expr, x):
        feats.forms.add("composite")
    if _is_rational_in_trig(expr, x):
        feats.forms.add("rational_in_sin_cos")

    # ---- 组合信号
    factors = list(expr.as_ordered_factors()) if isinstance(expr, sp.Mul) else [expr]
    terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]

    def any_factor(pred) -> bool:
        return any(pred(f) for f in factors + terms)

    def is_poly_factor(f) -> bool:
        d = _poly_degree_in(f, x)
        return bool(d and d >= 1)

    has_exp_factor = any_factor(lambda f: f.has(sp.exp) or _contains_func(f, (sp.exp,)))
    has_trig_factor = any_factor(lambda f: _contains_func(f, TRIG_FUNCS))
    has_log_factor = any_factor(lambda f: f.has(sp.log))
    has_invtrig_factor = any_factor(lambda f: _contains_func(f, INV_TRIG_FUNCS))

    # 指数/三角的自变量必须是**一次式**,才谈得上"多项式 × 指数"。
    # x·e^{x²} 虽然有"多项式 × 指数"的外形,但正确方法是凑微分
    # (因为 e^{x²} 的导数带出那个 x),把它推荐去分部积分是错的。
    has_exp_linear = _any_linear_arg(_exp_args(expr), x)
    has_trig_linear = _any_linear_arg(_trig_args(expr), x)

    if any_factor(is_poly_factor):
        # 多项式只作为**因子**出现时(如 x·eˣ),前面的整体判定拿不到它,
        # 这里补上,否则「多项式×指数」这类卡片会丢掉 poly 这一半的匹配。
        feats.families.add("poly")
        if has_exp_linear:
            feats.misc.add("poly_times_exp")
        if has_trig_linear:
            feats.misc.add("poly_times_trig")
        if has_log_factor:
            feats.misc.add("poly_times_log")
        if has_invtrig_factor:
            feats.misc.add("poly_times_invtrig")

    if has_exp_factor and has_trig_factor and not any_factor(is_poly_factor):
        feats.misc.add("cyclic_parts")

    sc = _sin_cos_power_info(expr, x)
    if sc:
        m, n = sc
        # m+n ≥ 2 这个限制是为了排除退化的情形:单独一个 cos x 虽然
        # 形式上满足"n 为奇数",但它就是基本积分表里的一条,
        # 不该被推荐去"拆一个 cos x 凑微分"。
        if m + n >= 2 and ((m % 2 == 1) or (n % 2 == 1)):
            feats.misc.add("odd_power_sin_cos")
        if m + n >= 2 and m % 2 == 0 and n % 2 == 0:
            feats.misc.add("even_power_sin_cos")

    if _is_trig_product_to_sum(expr, x):
        feats.misc.add("trig_product")

    if _is_rational_in_exp(expr, x):
        feats.misc.add("exponential_rational")

    if is_rational_fn and degree is None:
        try:
            num, den = sp.fraction(sp.cancel(expr))
            dn = _poly_degree_in(num, x) or 0
            dd = _poly_degree_in(den, x)
            if dd and dn >= dd:
                feats.misc.add("improper_fraction")
            if dd:
                for factor, mult in sp.factor(den).as_powers_dict().items():
                    if mult >= 2:
                        feats.misc.add("repeated_root")
                    if _poly_degree_in(factor, x) == 2:
                        try:
                            disc = sp.discriminant(sp.Poly(factor, x))
                            if disc.is_negative:
                                feats.misc.add("irreducible_quadratic")
                        except Exception:
                            pass
        except Exception:
            pass

    if _binomial_power_match(expr, x):
        feats.misc.add("binomial_power")

    return feats


# ---------------------------------------------------------------- 相似度
def structural_similarity(a: Counter, b: Counter) -> float:
    """两个节点直方图的余弦相似度(只用 type 名,不含具体系数)。"""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ================================================================ 定积分
# 需要小心的函数族:对含这些函数的表达式做 sp.simplify 会尝试大量变换,
# 有时候会慢到卡住。实测 ∫_0^{π/2} sin x/(sin x+cos x) dx 的特征提取就死在了
# sp.simplify(sin(x) + cos(x) - x) 上 —— 那只是在问"分母是不是恰好等于 x"。
_TRANSCENDENTAL = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log,
                   sp.sinh, sp.cosh, sp.asin, sp.atan)


def _zero_diff(a, b) -> bool:
    """判断两个表达式是否相等,但**不做昂贵的化简**。

    先看结构上是否直接相等;差值里含超越函数就直接判否 ——
    这类差值几乎不可能恒等于 0(真恒等的话结构上早就化简掉了),
    而 sp.simplify 在它们身上可能耗掉几分钟。
    """
    try:
        diff = sp.sympify(a) - sp.sympify(b)
    except Exception:
        return False
    if diff == 0:
        return True
    try:
        if diff.free_symbols and diff.has(*_TRANSCENDENTAL):
            return False
        return sp.simplify(diff) == 0
    except Exception:
        return False


def numeric_real_point(point) -> bool:
    """这个对象是不是一个"明确的实数点",能拿去和别的数比大小?

    必须挡掉 sp.singularities 对周期函数返回的东西 ——
    它给的是 ImageSet 无穷族(如 3π/4 + nπ 或 π/2 + nπ)。
    那种对象既不能枚举也不能比较大小,一旦混进 `lo < point < hi`
    就会卡死在符号比较上(实测 sin x/(sin x+cos x) 的特征提取就死在这里)。
    """
    try:
        if not isinstance(point, sp.Expr):
            return False
        if point.free_symbols:
            return False
        return bool(point.is_number) and bool(point.is_real)
    except Exception:
        return False


def explicit_singular_points(f: sp.Expr, x: sp.Symbol, max_depth: int = 4) -> list:
    """从 sp.singularities 的结果里挑出**明确的数值点**,其余一律丢弃。

    关键细节:只能走 `.args` 递归,**不能 `for p in result`**。
    sp.singularities 对周期函数返回的是 Union(ImageSet(...), ImageSet(...)),
    对无限集合做迭代会触发枚举,直接死循环 —— 而且守卫代码根本没机会执行,
    因为卡的是迭代本身,不是守卫。
    """
    try:
        result = sp.singularities(f, x)
    except Exception:
        return []

    out: list = []

    def visit(node, depth: int) -> None:
        if depth > max_depth:
            return
        if numeric_real_point(node):
            out.append(node)
            return
        if isinstance(node, (sp.FiniteSet, sp.Union, sp.Tuple, tuple, list)):
            args = node.args if hasattr(node, "args") else node
            for arg in args:
                visit(arg, depth + 1)

    visit(result, 0)
    return out


def _classify_interval(lo, hi) -> set[str]:
    """把区间归类。定积分的方法很大程度上由区间决定。"""
    tags: set[str] = set()
    try:
        lo, hi = sp.sympify(lo), sp.sympify(hi)
    except Exception:
        return {"any"}

    eq = _zero_diff

    infinite = lo in (sp.oo, -sp.oo) or hi in (sp.oo, -sp.oo)
    if infinite:
        tags.add("infinite")

    # 对称区间:[-a, a](含 (−∞, +∞) 与符号上限)
    if lo == -sp.oo and hi == sp.oo:
        tags.add("symmetric")
    elif not infinite:
        try:
            if eq(lo + hi, 0) and not eq(lo, 0):
                tags.add("symmetric")
        except Exception:
            if lo == -hi and lo != 0:
                tags.add("symmetric")

    if eq(lo, 0):
        if eq(hi, sp.oo):
            tags.add("zero_to_inf")
        if eq(hi, sp.pi / 2):
            tags.add("zero_to_pi_over_2")
        if eq(hi, sp.pi):
            tags.add("zero_to_pi")
        if eq(hi, 2 * sp.pi):
            tags.add("zero_to_two_pi")
        if eq(hi, 1):
            tags.add("unit")

    if not tags:
        tags.add("finite")
    return tags


def _unbounded_at(f: sp.Expr, x: sp.Symbol, point) -> bool:
    """f 在 x → point 附近是否无界(用来识别瑕点)。

    用**数值探针**而不是 sp.limit:limit 在某些表达式上会卡很久,
    而我们只需要一个"是不是趋向无穷"的判断。阈值取 100:
    x^{-1/3} 在 1e-9 处是 1000、x^{-1/2} 是 3.2e4、1/√(1−x²) 在端点附近是 2e4,
    都能抓到;而 log x 这类对数奇点(|值|只有几十)抓不到 —— 那个由
    log_singularity 这条单独的规则处理。
    阈值原来取的 1e4,结果漏掉了 x^{-1/3} 这种"温和的无界",导致后面的
    截断序列被端点奇点带偏、白白报"无法判定"(而符号层其实已经给出了正确的 π)。
    """
    try:
        point = sp.sympify(point)
        if point in (sp.oo, -sp.oo) or not point.is_number:
            return False
        func = sp.lambdify(x, f, modules=["mpmath"])
        base = complex(sp.N(point, 25))
    except Exception:
        return False
    for eps in (1e-6, 1e-9):
        for direction in (1, -1):
            try:
                value = func(base + direction * eps)
            except Exception:
                continue
            if value is None:
                continue
            try:
                if abs(value) > 100:
                    return True
            except Exception:
                return True
    return False


def _decay_kind(f: sp.Expr, x: sp.Symbol) -> None | str:
    """无穷远处的衰减方式:指数衰减还是代数衰减。"""
    try:
        if f.has(sp.exp):
            for node in sp.preorder_traversal(f):
                if isinstance(node, sp.exp):
                    arg = node.args[0]
                    try:
                        poly = sp.Poly(sp.expand(arg), x)
                    except Exception:
                        continue
                    if poly.degree() == 1 and poly.coeff_monomial(x).is_negative:
                        return "exponential_decay"
            return None
    except Exception:
        pass
    # 有理式:看分母次数是否高于分子
    try:
        if f.is_rational_function(x):
            num, den = sp.fraction(sp.cancel(f))
            dn, dd = sp.degree(num, x), sp.degree(den, x)
            if dd is not None and dn is not None and dd > dn:
                return "algebraic_decay"
    except Exception:
        pass
    return None


def extract_definite(f: sp.Expr, x: sp.Symbol, lo, hi) -> Features:
    """定积分的结构签名 = 不定积分的签名 + 区间类型 + 定积分特有的信号。"""
    feats = extract(f, x)
    feats.interval = _classify_interval(lo, hi)

    # 绝对值/取整这类不可导结构
    if f.has(sp.Abs, sp.Min, sp.Max, sp.sign, sp.floor, sp.ceiling):
        feats.misc.add("abs_or_floor")
        feats.families.add("abs")

    # 参数
    if any(s is not x and s.is_real for s in f.free_symbols):
        feats.misc.add("parameterized")

    # 振荡
    if f.has(sp.sin, sp.cos):
        feats.misc.add("oscillatory")
    if _is_trig_product_to_sum(f, x):
        feats.misc.add("product_of_trig")

    # 衰减方式
    kind = _decay_kind(f, x)
    if kind:
        feats.misc.add(kind)

    # 瑕点
    if _unbounded_at(f, x, lo) or _unbounded_at(f, x, hi):
        feats.misc.add("singular_endpoint")
    if hi != sp.oo and lo != -sp.oo:
        for point in _interior_candidates(f, x, lo, hi):
            if _unbounded_at(f, x, point):
                feats.misc.add("singular_interior")
                break
    if f.has(sp.log) and (lo == 0 or _unbounded_at(f, x, lo)):
        feats.misc.add("log_singularity")

    # 奇偶性
    try:
        negated = f.subs(x, -x)
        if _zero_diff(negated, -f) or _zero_diff(negated, f):
            feats.misc.add("even_odd")
    except Exception:
        pass

    # 标准形状
    _tag_standard_shapes(f, x, lo, hi, feats)
    return feats


def _interior_candidates(f: sp.Expr, x: sp.Symbol, lo, hi, samples: int = 200) -> list:
    """在 (lo, hi) 内部找可疑的奇点。

    两条路并用:
      · sp.singularities 给出的**明确数值点**(ImageSet 那类无穷族直接跳过)
      · 数值扫描分母的零点 —— 这条能补上周期型奇点,
        比如 tan x 在 [0,π] 上的 π/2,sp.singularities 只给出 π/2 + nπ。
    """
    import mpmath as mp  # 局部导入,避免顶层多一个依赖

    found: list = []
    lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
    for point in explicit_singular_points(f, x):
        try:
            if bool(lo_s < point < hi_s):
                found.append(sp.nsimplify(point))
        except Exception:
            continue

    try:
        denominator = sp.denom(sp.together(f))
        if denominator != 1 and denominator.has(x):
            func = sp.lambdify(x, denominator, modules=["mpmath"])
            with mp.workdps(25):
                a = mp.mpf(str(sp.N(lo_s, 25)))
                b = mp.mpf(str(sp.N(hi_s, 25)))
                step = (b - a) / samples
                for i in range(1, samples):
                    t = a + i * step
                    try:
                        value = complex(func(t))
                    except Exception:
                        continue
                    if abs(value) < 1e-6:
                        found.append(sp.N(t, 20))
    except Exception:
        pass
    return found


def _scaled_same_function(num: sp.Expr, x: sp.Symbol):
    """判断 num 是不是 `c·[f(a·x) − f(b·x)]`(a≠b)的形状,即 Frullani 的分子。

    认出时返回 (函数头, a, b);否则返回 None。

    ## 原来这段判据是**错的**

    旧代码写的是 `simplify(terms[0] + terms[1]) == 0` —— 意思是"两项之和为零",
    也就是要求**分子恒等于 0**。可真正的 Frullani 分子是
    "同一个函数在两个缩放下的**差**",不是"互为相反数"。
    所以 `frullani_shape` 这个 token 从来没被触发过(README §9 记过这处死代码)。
    顺带一提,那个 `simplify` 也是白留的一处卡顿点 —— 现在不需要了。

    ## 判据

    * 分子恰好两项,符号一正一负;
    * 两项的**数值系数绝对值相等**(否则不是同一族的差);
    * 两项是同一个函数头(exp / sin / cos / atan / log / 倒数…);
    * 两个自变量都只是 x 的**常数倍**(`inner/x` 不含 x)—— 排除了 f(ax+b);
    * 两个缩放系数不相等(相等则分子恒为 0)。
    """
    terms = sp.Add.make_args(num)
    if len(terms) != 2:
        return None

    def split(term):
        """把一项拆成 (数值系数, 函数头, 自变量);不是"函数套自变量"就返回 None。"""
        coefficient, rest = term.as_coeff_Mul()
        if not coefficient.is_number:
            return None
        if rest.func is sp.exp:                       # e^{u}
            return coefficient, sp.exp, rest.args[0]
        if isinstance(rest, sp.Function) and len(rest.args) == 1:
            return coefficient, rest.func, rest.args[0]   # sin(u) / atan(u) / log(u)…
        if isinstance(rest, sp.Pow) and rest.exp == -1:
            return coefficient, "recip", rest.base        # 1/u
        return None

    positive = [t for t in terms if t.as_coeff_Mul()[0].is_positive]
    negative = [t for t in terms if not t.as_coeff_Mul()[0].is_positive]
    if len(positive) != 1 or len(negative) != 1:
        return None

    left = split(positive[0])
    right = split(-negative[0])                        # 取负号,变成与左边同号
    if left is None or right is None:
        return None
    coefficient_l, head_l, inner_l = left
    coefficient_r, head_r, inner_r = right
    if head_l != head_r or coefficient_l != coefficient_r:
        return None

    try:
        scale_l = sp.simplify(inner_l / x)
        scale_r = sp.simplify(inner_r / x)
    except Exception:
        return None
    # 自变量必须只是 x 的常数倍(排掉 f(ax+b) 这种),且两倍率不同
    if scale_l.has(x) or scale_r.has(x) or scale_l == scale_r:
        return None
    return head_l, scale_l, scale_r


def _tag_standard_shapes(f: sp.Expr, x: sp.Symbol, lo, hi, feats: Features) -> None:
    """识别几种有专属方法的"标准形状"。

    这几个信号直接对应方法卡片:Gamma 形状想到 Γ(s)、Beta 形状想到 B(p,q)、
    Frullani 形状想到"两项之差除以 x"、正交性想到 ∫sin(mx)cos(nx)=0。
    一律用 _zero_diff 而不是 sp.simplify 做相等判断 —— 见它的注释。
    """
    eq = _zero_diff

    # Gamma 形状:x^p·e^{-cx},区间 [0,∞)
    if eq(lo, 0) and eq(hi, sp.oo):
        factors = f.as_ordered_factors() if isinstance(f, sp.Mul) else [f]
        has_x_power = any(isinstance(fa, sp.Pow) and fa.base == x for fa in factors)
        has_exp = any(fa.has(sp.exp) or (isinstance(fa, sp.Pow) and fa.base is sp.E)
                      for fa in factors)
        if has_x_power and has_exp:
            feats.misc.add("gamma_shape")
        # 幂分母形状:1/(1+x^n) 这类
        try:
            num, den = sp.fraction(sp.cancel(f))
            if num.is_number and den.has(x):
                feats.misc.add("power_denominator")
        except Exception:
            pass

    # Beta 形状:x^p·(1−x)^q,区间 [0,1]
    if eq(lo, 0) and eq(hi, 1):
        factors = f.as_ordered_factors() if isinstance(f, sp.Mul) else [f]
        has_x_power = any(isinstance(fa, sp.Pow) and fa.base == x for fa in factors)
        has_one_minus_x = any(
            isinstance(node, sp.Add) and _zero_diff(node, 1 - x)
            for node in sp.preorder_traversal(f))
        if has_x_power and has_one_minus_x:
            feats.misc.add("beta_shape")

    # Frullani 形状:c·[f(a·x) − f(b·x)] / x(判别见 _scaled_same_function)
    #
    # ★ 这里**不能**用 `sp.together(f)` 再取分子分母。那一步会把
    #   `exp(-a x)` 当成 `1/exp(a x)` 通分,于是
    #       (e^{-ax} − e^{-bx})/x   →   (−e^{ax} + e^{bx}) / (x·e^{ax}·e^{bx})
    #   分母再也不是 x,判据直接失败。这是这处死代码的**第二个**独立原因
    #   (第一个是判据本身写反了,见 _scaled_same_function)。
    #   改成在**因子层**找那个 `1/x`,剩下的因子乘积就是分子。
    try:
        factors = list(sp.Mul.make_args(f))
        reciprocals = [fa for fa in factors
                       if isinstance(fa, sp.Pow) and fa.base == x and fa.exp == -1]
        if len(reciprocals) == 1:
            rest = [fa for fa in factors if fa is not reciprocals[0]]
            numerator = sp.expand(sp.Mul(*rest)) if rest else sp.Integer(1)
            if _scaled_same_function(numerator, x) is not None:
                feats.misc.add("frullani_shape")
    except Exception:
        pass

    # 三角正交性:区间是一个整周期,且含两个不同频率的三角函数
    if eq(lo, 0) and (eq(hi, sp.pi) or eq(hi, 2 * sp.pi)):
        freqs = set()
        for node in sp.preorder_traversal(f):
            if isinstance(node, (sp.sin, sp.cos)) and len(node.args) == 1:
                try:
                    freqs.add(sp.Poly(sp.expand(node.args[0]), x).coeff_monomial(x))
                except Exception:
                    pass
        if len(freqs) >= 2:
            feats.misc.add("trig_orthogonality")
