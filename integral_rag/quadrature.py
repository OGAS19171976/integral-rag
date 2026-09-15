"""定积分的数值求值与敛散性判定 —— 这是定积分验证链的真值来源。

## 为什么需要它

不定积分的验证靠"求导回验";定积分没有导数可求,但它有一个**更强**的真值来源:
高精度数值求积。定积分的结果就是一个数,数值算出来是另一个数,对得上就是对得上 ——
这比"逐点恒等式"更硬,而且顺便就把敛散性判了。

## 但数值求积本身很容易骗人

实测 mpmath 的 `quad` 单独使用会给出这些结果:

    ∫_1^∞ dx/x          发散,却返回 125          ← 会假阳性!
    ∫_0^∞ sin(x)/x dx   应为 π/2,却返回 8.88
    ∫_0^∞ cos(x²) dx    返回 3.9e53
    ∫_{-1}^{2}|x| dx    Abs 的折点没切开,精度掉到 2.49999

所以这里不是"调一下 quad",而是把一整套策略组合起来:

    ① 在折点(Abs/floor 的拐点)与极点处**把区间切开**
    ② 端点奇点直接交给 tanh-sinh —— 它天生处理这个(实测 x^{-3/4} 能到 20 位)
    ③ 无穷限用**截断序列看趋势**,而不是一次 quad 到底
    ④ 线性频率的振荡用 quadosc
    ⑤ x^p 型加速振荡先换元成线性频率,再交给 quadosc
    ⑥ 尽量用两种独立方法交叉验证;不一致就报"无法判定",绝不给一个可疑的数

## 三种结论,不含糊

    convergent(value)  收敛,并且给出了可信的值
    divergent          发散(附上判据,比如"部分积分按 ln R 增长")
    inconclusive       数值上判不了 —— 调用方必须据此**放弃**验证,
                       而不是当成"验证失败"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mpmath as mp
import sympy as sp

from .features import explicit_singular_points

_TRIG = (sp.sin, sp.cos)
_DEFAULT_DPS = 40

# 收敛序列的判据:相邻两项的相对变化要小于这个值
_SETTLE_TOL = 1e-6
# 认为"增长到无穷"的判据:末项比首项大这么多倍
_GROWTH_FACTOR = 50.0


@dataclass
class NumericResult:
    verdict: str                       # convergent / divergent / inconclusive
    value: complex | None = None
    method: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "convergent"

    @property
    def divergent(self) -> bool:
        return self.verdict == "divergent"

    def describe(self) -> str:
        head = {"convergent": "收敛", "divergent": "发散", "inconclusive": "无法判定"}[self.verdict]
        if self.value is not None:
            head += f" ≈ {_fmt(self.value)}"
        return f"{head}({self.method})"


def _fmt(value, digits: int = 16) -> str:
    """把 mpf / mpc / Python 数都格式化成短字符串。

    注意 mpf 不认 f-string 的格式说明符(`format(mpf, '.16g')` 会 TypeError),
    必须先转成 Python 的 complex。
    """
    if value is None:
        return "?"
    try:
        as_complex = complex(value)
    except Exception:
        return str(value)
    if abs(as_complex.imag) > 1e-30:
        return f"{as_complex.real:.{digits}g}{as_complex.imag:+.{digits}g}i"
    return f"{as_complex.real:.{digits}g}"


# ================================================================ 基础工具
def _to_mpf(value, dps: int = _DEFAULT_DPS):
    """SymPy 的数(含 pi、oo)转成高精度 mpf。

    注意:上下限常常是 Python 原生的 int(0、1),不是 SymPy 对象,
    所以必须先 sympify —— 否则 `value.is_Rational` 会直接 AttributeError。
    """
    value = sp.sympify(value)
    if value == sp.oo:
        return mp.inf
    if value == -sp.oo:
        return -mp.inf
    with mp.workdps(dps + 10):
        if value.is_Rational:
            return mp.mpf(int(value.p)) / mp.mpf(int(value.q))
        return mp.mpf(str(sp.N(value, dps + 10)))


def _lambdify(f: sp.Expr, x: sp.Symbol):
    """SymPy 表达式 → mpmath 可调用对象。"""
    return sp.lambdify(x, f, modules=["mpmath"])


def _guarded(func):
    """把端点上偶发的 nan/inf 当成 0。

    这是标准的工程处理:出问题的只是个别测度为零的端点(比如 log x 在 x=0),
    而那个点上的取值对积分没有贡献。

    但**必须设一个失败次数上限**。否则会酿成一个很危险的事故:
    如果整个函数根本没法求值(典型的例子是参数没代进去,lambdify 出来
    需要两个参数的函数),那么每次调用都抛异常、每次都被当成 0,
    于是积分看起来恒等于 0 —— 数值层会兴高采烈地报告"收敛 ≈ 0"。
    实测 ∫_0^∞ e^{-tx}dx 在没代 t 的时候就掉进了这个坑。

    超过上限就重新抛出,让 mp.quad 失败,由上层如实报"无法判定"。
    """
    state = {"failures": 0}

    def wrapped(t):
        try:
            value = func(t)
        except Exception:
            state["failures"] += 1
            if state["failures"] > 5:
                raise
            return mp.mpf(0)
        if value is None:
            state["failures"] += 1
            if state["failures"] > 5:
                raise ValueError("被积函数无法求值")
            return mp.mpf(0)
        try:
            if mp.isnan(value) or mp.isinf(value):
                return mp.mpf(0)
        except Exception:
            pass
        return value

    return wrapped


def _safe(call, *args, **kwargs):
    """调用 mpmath 函数并吞掉异常/nan,返回 (成功?, 值)。

    必须转发 kwargs —— quadosc 的 omega 就是关键字参数。
    """
    try:
        value = call(*args, **kwargs)
    except Exception:
        return False, None
    if value is None:
        return False, None
    try:
        if mp.isnan(value) or mp.isinf(value):
            return False, None
    except Exception:
        return False, None
    return True, value


def _quad(func, points) -> tuple[bool, object]:
    """在给定端点列表上求积(端点可以含 ±inf)。"""
    return _safe(mp.quad, _guarded(func), points)


# ================================================================ 去奇点 / 折点
def _breakpoints(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> list[sp.Expr]:
    """找出区间内部需要切开的位置:极点、以及 Abs/floor 的折点。

    这是效果最明显的一步:∫_{-1}^{2}|x|dx 整体 quad 得到 2.49999,
    切成 [-1,0]+[0,2] 立刻就是精确的 2.5。
    """
    lo_num, hi_num = lo, hi
    candidates: set[sp.Expr] = set()

    # 极点:分母的零点
    try:
        denom = sp.denom(sp.together(f))
        if denom != 1:
            for root in sp.solve(sp.Eq(denom, 0), x):
                candidates.add(root)
    except Exception:
        pass

    # 奇异点(SymPy 自己认得的那些)。走 explicit_singular_points 而不是直接迭代:
    # 对周期函数 sp.singularities 返回的是 Union(ImageSet(...)),迭代它会枚举
    # 无限集合,直接死循环。
    for point in explicit_singular_points(f, x):
        candidates.add(point)

    # 折点:Abs / Min / Max / sign / floor 的变号点或分段点
    for node in sp.preorder_traversal(f):
        if isinstance(node, sp.Abs) and node.args:
            try:
                for root in sp.solve(sp.Eq(node.args[0], 0), x):
                    candidates.add(root)
            except Exception:
                pass
        elif isinstance(node, (sp.Min, sp.Max, sp.sign, sp.floor, sp.ceiling)):
            for arg in node.args:
                try:
                    for root in sp.solve(sp.Eq(arg, 0), x):
                        candidates.add(root)
                except Exception:
                    pass

    inside: list[sp.Expr] = []
    for point in candidates:
        if not point.is_real:
            continue
        try:
            if point.is_number and lo_num.is_number and hi_num.is_number:
                if bool(lo_num < point < hi_num):
                    inside.append(sp.nsimplify(point))
        except Exception:
            continue
    # 去掉重复与数值上极近的点
    unique: list[sp.Expr] = []
    for point in sorted(inside, key=lambda p: float(sp.N(p, 20))):
        if not unique or abs(float(sp.N(point - unique[-1], 20))) > 1e-12:
            unique.append(point)
    return unique


# ================================================================ 振荡识别
def _oscillation(f: sp.Expr, x: sp.Symbol):
    """识别振荡结构。返回 (kind, omega, power)。

    kind = "linear"   → sin/cos(ωx+φ),频率恒定 → 可以用 quadosc
    kind = "power"    → sin/cos(ωx^p+φ), p>1,频率递增 → 先换元再 quadosc
    kind = None       → 不振荡
    """
    linear_omegas: list[sp.Expr] = []
    power_info = None
    for node in sp.preorder_traversal(f):
        if not (isinstance(node, _TRIG) and len(node.args) == 1):
            continue
        arg = sp.expand(node.args[0])
        try:
            poly = sp.Poly(arg, x)
        except Exception:
            continue
        deg = poly.degree()
        if deg == 1:
            coeff = poly.coeff_monomial(x)
            if coeff.is_number and coeff != 0:
                linear_omegas.append(abs(coeff))
        elif deg is not None and deg > 1 and poly.coeff_monomial(x) == 0:
            # 只处理单项 x^p 的情形(如 cos(x²))
            top = poly.coeff_monomial(x**deg)
            if top.is_number and top != 0:
                power_info = (abs(top), int(deg))
    if linear_omegas:
        return "linear", min(linear_omegas), None
    if power_info is not None:
        return "power", power_info[0], power_info[1]
    return None, None, None


def _averaged_abs(func, base: float, window_ratio: float = 1.0) -> float | None:
    """在 [base, base·(1+ratio)] 上取 |f| 的平均值。振荡会被自然平滑掉。"""
    steps = 400
    with mp.workdps(_DEFAULT_DPS):
        start = mp.mpf(base)
        width = start * window_ratio
        step = width / steps
        total = mp.mpf(0)
        for i in range(steps):
            ok, value = _safe(func, start + i * step)
            if not ok:
                return None
            try:
                total += abs(value)
            except Exception:
                return None
        return total / steps


def _decays_at_infinity(f: sp.Expr, x: sp.Symbol, kind: str):
    """无穷远处 |f| 是否趋于 0(振荡时按窗口平均)。

    注意:这个判据只对**频率恒定**的振荡有意义。cos(x²) 这类加速振荡的
    平均值并不趋于 0,但它的积分是收敛的(靠越来越快的相消),
    所以那种情形要跳过这个判据,不能据此判发散。
    """
    if kind == "power":
        return None
    func = _lambdify(f, x)
    samples = []
    for base in (100, 1000, 10**4, 10**5):
        value = _averaged_abs(func, base)
        if value is None:
            return None
        samples.append(value)
    notes = samples
    # 单调下降且末项已经很小 → 认为在衰减
    decreasing = all(notes[i + 1] < notes[i] * 0.9 for i in range(len(notes) - 1))
    small = notes[-1] < 1e-3
    return bool(decreasing and small)


# ================================================================ 截断序列
def _truncation_sequence(f: sp.Expr, x: sp.Symbol, start, rs) -> list[tuple[float, object]]:
    func = _lambdify(f, x)
    out = []
    for R in rs:
        ok, value = _quad(func, [start, mp.mpf(R)])
        out.append((float(R), value if ok else None))
    return out


def _classify_sequence(values: list) -> str:
    """把部分积分序列分成 converging / growing / oscillating / unknown。

    关键是认出"对数发散"这种形状:∫_1^R dx/x = ln R 的序列
    [2.30, 4.61, 6.91, 9.21, …] 单项看着都很温和,末项除以中位数也才 1.5 倍,
    用"爆炸式增长"的判据完全抓不住。它的特征是:**单调递增而且增量不收缩**。
    """
    good = [v for v in values if v is not None]
    if len(good) < 3:
        return "unknown"

    try:
        complex_values = [complex(v) for v in good]
    except Exception:
        return "unknown"
    if any(abs(c.imag) > 1e-20 for c in complex_values):
        return "unknown"

    vals = [c.real for c in complex_values]
    mags = [abs(v) for v in vals]
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]

    # ⓪ 已经完全稳定(差值都在数值噪声以下)→ 收敛。
    # 指数衰减型(如 e^{-x²})的部分积分在 R=10 时就已经到机器精度了,
    # 后面的差值全是 0;如果不先判这一条,会掉进"差值不收缩"的坑里被判成振荡。
    if diffs and max(abs(d) for d in diffs) <= 1e-12 * max(1.0, max(mags)):
        return "converging"

    # ① 单调增长且增量不收缩 → 发散(对数发散、幂发散都属于这一类)
    non_decreasing = all(d >= -1e-12 * max(1.0, mags[i]) for i, d in enumerate(diffs))
    # 参考增量要取**第一个非零**的:当截断点是从 0 开始时前几项会恰好是 0,
    # 用 diffs[0]=0 做基准会让"9e-6 >= 0.3×0"恒成立,把收敛误判成发散。
    first_nonzero = next((abs(d) for d in diffs if abs(d) > 1e-12), 0.0)
    if non_decreasing and first_nonzero > 0 and abs(diffs[-1]) > 1e-9:
        if abs(diffs[-1]) >= 0.3 * first_nonzero and mags[-1] > 1.0:
            return "growing"

    # ② 爆炸式增长 → 发散(振荡积分的部分积分失控时是这个形状)
    if len(mags) >= 2 and mags[-1] > max(mags[:-1]) * _GROWTH_FACTOR:
        return "growing"

    # ③ 差值收缩 → 收敛
    if diffs:
        tail = abs(diffs[-1])
        scale = max(1.0, mags[-1])
        if tail <= scale * 1e-4 and tail < first_nonzero * 0.5:
            return "converging"
        shrinking = all(abs(diffs[i + 1]) < abs(diffs[i]) for i in range(len(diffs) - 1))
        if shrinking and tail <= scale * 1e-3:
            return "converging"

    return "oscillating"


# ================================================================ 有限区间
def _endpoint_blows_up(f: sp.Expr, x: sp.Symbol, endpoint, side: str) -> bool:
    """被积函数在端点附近是否无界。"""
    func = _lambdify(f, x)
    with mp.workdps(_DEFAULT_DPS):
        eps = mp.mpf(10) ** -12
        probe = endpoint + eps if side == "lower" else endpoint - eps
    ok, value = _safe(func, probe)
    if not ok:
        return True
    try:
        return abs(value) > mp.mpf(10) ** 5
    except Exception:
        return True


def _epsilon_sequence(f: sp.Expr, x: sp.Symbol, endpoint, other, side: str):
    """从奇点侧逐步逼近,看 ∫ 是否收敛。

    用于端点是一阶极点这类 tanh-sinh 也吃不动的情形:
    ∫_ε^1 dx/x = −ln ε → ε 越小值越大,一眼看出是发散。
    而 ∫_ε^1 x^{-1/2}dx = 2(1−√ε) → 有界收敛,不会被误判。
    """
    func = _lambdify(f, x)
    eps_list = [mp.mpf(10) ** -k for k in (2, 4, 6, 8)]
    values = []
    for eps in eps_list:
        if side == "lower":
            points = [endpoint + eps, other]
        else:
            points = [other, endpoint - eps]
        ok, value = _quad(func, points)
        values.append(value if ok else None)
    good = [v for v in values if v is not None]
    if len(good) < 3:
        return None, "unknown"
    state = _classify_sequence(good)
    if state == "growing":
        return None, "growing"
    if state in ("converging", "oscillating"):
        # 用最后两个值做一次简单的极限外推
        return good[-1], state
    return None, state


def _finite_integral(f: sp.Expr, x: sp.Symbol, lo, hi, notes: list[str]) -> NumericResult:
    interior = _breakpoints(f, x, lo, hi)
    func = _lambdify(f, x)
    lo_mpf, hi_mpf = _to_mpf(lo), _to_mpf(hi)

    if interior:
        notes.append(f"在 {len(interior)} 个内部点处切开区间:{interior}")
    bounds = [lo] + interior + [hi]

    total = None
    for left, right in zip(bounds, bounds[1:]):
        # 先查端点上是不是一阶极点。
        # 必须查:tanh-sinh 遇到端点极点不会报错,而是**给出一个很大的有限数** ——
        # ∫_0^1 dx/x 会被算成 102.02。这种"看起来很正常的收敛值"最危险,
        # 会直接造成假阳性,所以一定要单独用逼近序列判一次。
        for endpoint, other, side in ((left, right, "lower"), (right, left, "upper")):
            if not sp.sympify(endpoint).is_number:
                continue
            if not _endpoint_blows_up(f, x, endpoint, side):
                continue
            _, state = _epsilon_sequence(f, x, endpoint, other, side)
            if state == "growing":
                return NumericResult(
                    "divergent", None, "epsilon-sequence",
                    notes + [f"在 x={endpoint} 附近部分积分无界增长(一阶极点),积分发散"])

        ok, value = _quad(func, [_to_mpf(left), _to_mpf(right)])
        if not ok:
            # 端点可能是更难处理的奇点,用逼近序列再试
            for endpoint, other, side in ((left, right, "lower"), (right, left, "upper")):
                if sp.sympify(endpoint) in (sp.oo, -sp.oo):
                    continue
                value2, state = _epsilon_sequence(f, x, endpoint, other, side)
                if state == "growing":
                    return NumericResult("divergent", None, "epsilon-sequence",
                                         notes + [f"在 x={endpoint} 附近部分积分无界增长"])
                if value2 is not None:
                    ok, value = True, value2
                    break
            if not ok:
                return NumericResult("inconclusive", None, "quad",
                                     notes + [f"[{left},{right}] 上求积失败"])
        total = value if total is None else total + value

    # 交叉验证:整体一次 quad。折点存在时它精度会差,只作为数量级核对
    if not interior:
        ok, whole = _quad(func, [lo_mpf, hi_mpf])
        if ok and abs(total) > 1e-12 and abs(whole - total) > 1e-6 * max(1.0, abs(total)):
            notes.append(f"整体求积 {_fmt(whole)} 与分段求积 {_fmt(total)} 不一致")
            return NumericResult("inconclusive", None, "quad", notes)

    return NumericResult("convergent", total, "tanh-sinh(分段)", notes)


# ================================================================ 无穷限
def _substitute_power_oscillation(f: sp.Expr, x: sp.Symbol, lo, hi, omega, power):
    """把 sin/cos(ωx^p) 通过 u = ωx^p 换成线性频率的振荡。

    ∫_0^∞ cos(x²)dx 直接求积会给出 3.9e53;换成
    ∫_0^∞ cos(u)/(2√u) du 之后,quadosc 就能给出 0.62…
    """
    u = sp.Symbol("_qu", positive=True)
    # x = (u/ω)^{1/p}
    x_of_u = (u / omega) ** (sp.Rational(1, power))
    new_f = sp.simplify(f.subs(x, x_of_u) * sp.diff(x_of_u, u))
    new_lo = sp.simplify(omega * lo**power) if lo != 0 else sp.Integer(0)
    new_hi = sp.oo if hi == sp.oo else sp.simplify(omega * hi**power)
    return new_f, u, new_lo, new_hi


def _quadosc_value(f: sp.Expr, x: sp.Symbol, lo, hi, omega) -> tuple[bool, object]:
    """振荡积分的值:先 quad 头部,再对尾巴用 quadosc。

    整体交给 quadosc 精度是不够的 —— 实测 ∫_0^∞ cos(u)/(2√u) du
    整体 quadosc 得到 0.62465,真值 0.626657,差 3e-3;
    但切成 [0,U] + [U,∞) 之后 quadosc 处理的是一个快速衰减的尾巴,
    精度立刻到 1e-20。U 取两个不同的值交叉验证,不一致就不要这个值。
    """
    if lo in (sp.oo, -sp.oo) or hi not in (sp.oo, -sp.oo):
        return False, None

    func = _lambdify(f, x)
    omega_mpf = mp.mpf(str(sp.N(omega, 20)))
    head_start = _to_mpf(lo)
    tail_end = _to_mpf(hi)

    results = []
    for U in (10, 60):
        head_ok, head = _quad(func, [head_start, mp.mpf(U)])
        tail_ok, tail = _safe(mp.quadosc, _guarded(func), [mp.mpf(U), tail_end], omega=omega_mpf)
        if head_ok and tail_ok:
            results.append(head + tail)

    if not results:
        ok, value = _safe(mp.quadosc, _guarded(func), [head_start, tail_end], omega=omega_mpf)
        return ok, value
    if len(results) == 2:
        agree, _ = close(complex(results[0]), complex(results[1]), rel_tol=1e-10)
        if not agree:
            return False, None
    return True, results[-1]


def _map_to_finite(f: sp.Expr, x: sp.Symbol, lo, hi):
    """把无穷区间用线性分式变换映到 [0,1],交给 tanh-sinh。

        ∫_a^∞ f dx  →  x = a + u/(1−u),dx = du/(1−u)²,u ∈ [0,1)

    为什么这一招很关键:代数衰减的尾巴(比如 x^{-1.5})在截断序列里收敛极慢
    —— R 得取到 1e5 以上才能把相对误差压到 1e-4,判据很容易判成"趋势不明"。
    但变换之后它在 u=1 处变成 (1−u)^{-1/2} 型的**可积奇点**,
    而这正是 tanh-sinh 的强项:实测能直接给到 20 位有效数字。
    """
    u = sp.Symbol("_mu", positive=True)
    try:
        # 还有未代值的参数时不要做这个变换:sp.simplify 碰到带符号指数的
        # 分式变换会陷进去出不来(实测 x^{s-1}/(1+x) 卡了半小时)。
        if f.free_symbols - {x}:
            return None
        if hi == sp.oo and lo != -sp.oo:
            x_of_u = sp.sympify(lo) + u / (1 - u)
        elif lo == -sp.oo and hi != sp.oo:
            x_of_u = sp.sympify(hi) - u / (1 - u)
        else:
            return None
        new_f = sp.simplify(f.subs(x, x_of_u) * sp.diff(x_of_u, u))
    except Exception:
        return None
    if new_f.has(sp.zoo, sp.nan) or new_f.free_symbols - {u}:
        return None
    return new_f, u, sp.Integer(0), sp.Integer(1)


def _improper_integral(f: sp.Expr, x: sp.Symbol, lo, hi, notes: list[str]) -> NumericResult:
    # 反向的无穷限(如 ∫_a^{−∞}):交换并取相反数,统一成"上限为 +∞"
    if hi == -sp.oo:
        swapped = _improper_integral(f, x, hi, lo, notes)
        if swapped.value is not None:
            return NumericResult(swapped.verdict, -swapped.value, f"flip → {swapped.method}",
                                 swapped.notes)
        return NumericResult(swapped.verdict, None, f"flip → {swapped.method}", swapped.notes)

    # 两侧都无穷:在 0 处拆开
    if lo == -sp.oo and hi == sp.oo:
        left = _improper_integral(f, x, -sp.oo, sp.Integer(0), notes)
        right = _improper_integral(f, x, sp.Integer(0), sp.oo, notes)
        if left.divergent or right.divergent:
            return NumericResult("divergent", None, "split-at-zero",
                                 notes + ["其中一个半轴发散"])
        if left.ok and right.ok:
            return NumericResult("convergent", left.value + right.value,
                                 "split-at-zero", notes)
        return NumericResult("inconclusive", None, "split-at-zero", notes)

    # (−∞, a]:换元 t = −x,变成 [−a, +∞)。
    # 注意端点顺序:∫_{−∞}^{a} f(x)dx = ∫_{−a}^{+∞} f(−t)dt —— 上下限一起翻。
    if lo == -sp.oo:
        t = sp.Symbol("_qt", real=True)
        flipped = f.subs(x, -t)
        inner = _improper_integral(flipped, t, -sp.sympify(hi), sp.oo, notes)
        if inner.ok or inner.divergent:
            return NumericResult(inner.verdict, inner.value, f"flip → {inner.method}", inner.notes)
        return NumericResult("inconclusive", None, "flip", notes)

    finite_end = lo if hi in (sp.oo, -sp.oo) else hi
    if finite_end in (sp.oo, -sp.oo):
        return NumericResult("inconclusive", None, "quad", notes + ["两端都不是有限值"])

    kind, omega, power = _oscillation(f, x)

    # ---- 非振荡型:首选"线性分式变换映到 [0,1] + tanh-sinh"
    # 代数衰减的尾巴在截断序列里收敛极慢(R 得取到 1e5 以上),变换之后它
    # 变成可积的端点奇点,tanh-sinh 直接给到 20 位。
    #
    # **必须排在振荡检测之后**:振荡积分做这个变换之后,
    # 在 u→1 处振荡频率趋于无穷(sin(u/(1−u))),tanh-sinh 完全失效 ——
    # 实测 ∫_0^∞ sin x/x dx 因为这一步被放到前面而回归成"未决"。
    if kind is None:
        mapped = _map_to_finite(f, x, lo, hi)
        if mapped is not None:
            new_f, u, new_lo, new_hi = mapped
            inner = _finite_integral(new_f, u, new_lo, new_hi, [])
            if inner.ok:
                notes.append(f"线性分式变换 x → u/(1−u) 后交给 tanh-sinh:∫{new_f} d{u}")
                return NumericResult("convergent", inner.value, "mapped-to-finite", notes)
            if inner.divergent:
                return NumericResult("divergent", None, "mapped-to-finite",
                                     notes + inner.notes)

    # ---- 加速振荡:先换元
    if kind == "power":
        try:
            new_f, u, new_lo, new_hi = _substitute_power_oscillation(f, x, lo, hi, omega, power)
            notes.append(f"先把 sin/cos(ωx^{power}) 换元成线性频率:∫{new_f} d{u}")
            inner = _improper_integral(new_f, u, new_lo, new_hi, notes)
            if inner.ok or inner.divergent:
                return NumericResult(inner.verdict, inner.value,
                                     f"power-substitution → {inner.method}", inner.notes)
            return NumericResult("inconclusive", None, "power-substitution", notes)
        except Exception as exc:
            notes.append(f"加速振荡换元失败:{type(exc).__name__}")

    # ---- 恒定频率振荡
    if kind == "linear":
        decay = _decays_at_infinity(f, x, kind)
        if decay is False:
            return NumericResult("divergent", None, "decay-test",
                                 notes + ["|f| 在无穷远处不趋于 0,振荡积分不可能收敛"])
        if decay is True:
            ok, value = _quadosc_value(f, x, lo, hi, omega)
            if ok:
                notes.append(f"quadosc(ω={omega}),头部 quad + 尾巴 quadosc 交叉验证通过")
                return NumericResult("convergent", value, "quadosc", notes)

    # ---- 通用路线:截断序列
    start = _to_mpf(finite_end)
    # 截断点必须**绝对**拉开:如果有限端点是 0,用 start*10^k 会永远是 0,
    # 序列退化成 [0,0,0,…],什么也判不出来(∫_0^∞ e^{-x}dx 就栽在这里)。
    base = max(abs(start), mp.mpf(1))
    scale_points = [base * mp.mpf(10) ** k for k in range(1, 7)]
    sequence = _truncation_sequence(f, x, start, scale_points)
    values = [v for _, v in sequence]
    state = _classify_sequence(values)
    notes.append("截断序列 " + "  ".join(
        _fmt(v) if v is not None else "失败" for v in values))

    if state == "growing":
        return NumericResult("divergent", None, "truncation-sequence",
                             notes + ["部分积分随截断上限无界增长"])
    if state == "converging":
        tail = [v for v in values if v is not None][-2:]
        if len(tail) == 2 and abs(tail[0]) > 1e-30:
            value = tail[1] + (tail[1] - tail[0]) / 9.0
        else:
            value = tail[-1]
        return NumericResult("convergent", value, "truncation-sequence", notes)
    if state == "oscillating":
        return NumericResult("inconclusive", None, "truncation-sequence",
                             notes + ["部分积分既不收敛也不明显发散(振荡未衰减)"])
    return NumericResult("inconclusive", None, "truncation-sequence", notes + ["趋势无法判断"])


# ================================================================ 主入口
def _smooth_endpoint(f: sp.Expr, x: sp.Symbol, lo, hi):
    """有限端点是瑕点时,换元把奇异抹平。

    实测 ∫_0^∞ dx/(√x(1+x)):被积函数在 0 处是 x^{-1/2} 型奇点,
    直接做截断序列会因为这种剧烈变化而判不出趋势(白白报"无法判定",
    而符号层明明给出了正确的 π)。令 x = u² 之后被积函数变成 2/(1+u²)
    —— 光滑、衰减快,截断序列一眼就收敛。

    返回 (新被积函数, 新变量, 新下限, 新上限),或者 None。
    """
    for endpoint, side, other in ((lo, "lower", hi), (hi, "upper", lo)):
        if endpoint in (sp.oo, -sp.oo):
            continue
        if not _endpoint_blows_up(f, x, endpoint, side):
            continue
        u = sp.Symbol("_su", positive=True)
        try:
            if side == "lower":
                x_of_u = sp.sympify(endpoint) + u**2
            else:
                x_of_u = sp.sympify(endpoint) - u**2
            new_f = sp.simplify(f.subs(x, x_of_u) * 2 * u)
        except Exception:
            continue
        if new_f.has(sp.zoo, sp.nan):
            continue
        # 两种情形换元后都是 [0, √(另一端−端点)];另一端无穷时就是 [0,∞)
        if other in (sp.oo, -sp.oo):
            new_hi = sp.oo
        else:
            try:
                new_hi = sp.sqrt(sp.simplify(sp.sympify(other) - sp.sympify(endpoint)))
            except Exception:
                continue
        return new_f, u, sp.Integer(0), new_hi
    return None


def evaluate(f: sp.Expr, x: sp.Symbol, a, b, params: dict | None = None,
             dps: int = _DEFAULT_DPS, _depth: int = 0) -> NumericResult:
    """数值求值 ∫_a^b f dx,同时判定敛散性。

    params 是参数的数值取值,例如 {"a": 2.0}。
    """
    notes: list[str] = []
    a, b = sp.sympify(a), sp.sympify(b)
    expr = f
    if params:
        # 参数名要按**符号名**去匹配:调用方通常传 {"a": 2.0} 这样的字符串键,
        # 而 f.free_symbols 里是带假设的 Symbol(a, positive=True),
        # 直接做 `"a" in f.free_symbols` 永远是 False,参数就悄悄没代进去。
        by_name: dict[str, sp.Symbol] = {}
        for source in (f.free_symbols, a.free_symbols, b.free_symbols):
            for sym in source:
                by_name.setdefault(sym.name, sym)
        subs = {}
        for key, value in params.items():
            name = key if isinstance(key, str) else getattr(key, "name", str(key))
            if name in by_name:
                subs[by_name[name]] = sp.Float(str(value), dps)
        if subs:
            expr = f.subs(subs)
            # 积分限里也可能带参数(比如 ∫_{-a}^{a} x²dx),必须一起代掉
            a = a.subs(subs)
            b = b.subs(subs)
            notes.append("参数取值 " + ", ".join(f"{k}={v}" for k, v in params.items()))

    # 代完之后积分限若还带符号,数值求积无从下手
    if not (a.is_number and b.is_number):
        return NumericResult("inconclusive", None, "symbolic-limits",
                             notes + [f"积分限仍含符号({a} → {b}),无法数值求积"])

    # 被积函数还有没代值的参数 → 直接说"判不了",不要往下走。
    # 这一条不是可有可无的:后面的换元/化简会拿**符号参数**去做
    # sp.simplify(比如把 x^{s−1}/(1+x) 变换到有限区间),
    # 那是一次可能几分钟到几十分钟不返回的符号运算。
    # 数值层本来就不该处理符号参数 —— 参数代值是调用方必须先做的事
    # (定积分那边会换成几组具体数值再来)。
    leftover = expr.free_symbols - {x}
    if leftover:
        return NumericResult("inconclusive", None, "symbolic-parameters",
                             notes + ["被积函数仍含未代值的参数 "
                                      + ", ".join(sorted(s.name for s in leftover))
                                      + ",无法数值求积"])

    # 有限端点是瑕点 → 先换元抹平,再递归求值。
    # 只做一层,避免反复换元把自己绕进去。
    if _depth == 0:
        smoothed = _smooth_endpoint(expr, x, a, b)
        if smoothed is not None:
            new_f, u, new_lo, new_hi = smoothed
            notes.append(f"端点有瑕点,先用 x = u² 型换元抹平:{new_f}")
            inner = evaluate(new_f, u, new_lo, new_hi, None, dps, _depth + 1)
            if inner.ok or inner.divergent:
                merged = list(dict.fromkeys(notes + inner.notes))
                return NumericResult(inner.verdict, inner.value,
                                     f"smooth-endpoint → {inner.method}", merged)

    if not isinstance(expr, sp.Expr):
        return NumericResult("inconclusive", None, "input", ["表达式无效"])

    lo, hi = a, b
    sign = 1
    try:
        if lo.is_number and hi.is_number and bool(lo > hi):
            lo, hi = hi, lo
            sign = -1
            notes.append("上下限已交换,结果取相反数")
    except Exception:
        pass

    # 上下限相等 → 0(约定)
    try:
        if sp.simplify(hi - lo) == 0:
            return NumericResult("convergent", 0, "trivial", notes)
    except Exception:
        pass

    infinite = lo in (sp.oo, -sp.oo) or hi in (sp.oo, -sp.oo)
    try:
        with mp.workdps(dps):
            result = (_improper_integral if infinite else _finite_integral)(
                expr, x, lo, hi, notes)
    except Exception as exc:
        return NumericResult("inconclusive", None, "exception",
                             notes + [f"{type(exc).__name__}: {exc}"])

    if sign == -1 and result.value is not None:
        result = NumericResult(result.verdict, -result.value, result.method, result.notes)
    # 合并而不是覆盖:内部分支可能已经把失败原因追加进了一个新列表
    merged = list(dict.fromkeys(list(notes) + list(result.notes)))
    result.notes = merged
    return result


def close(a: complex, b: complex, rel_tol: float = 1e-8, abs_tol: float = 1e-12) -> tuple[bool, float]:
    """两个数是否足够接近。返回 (是否接近, 相对误差)。"""
    try:
        diff = abs(complex(a) - complex(b))
    except Exception:
        return False, float("inf")
    scale = max(abs(complex(a)), abs(complex(b)), 1e-30)
    rel = diff / scale
    return (rel <= rel_tol or diff <= abs_tol), rel
