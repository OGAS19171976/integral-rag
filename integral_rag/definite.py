"""定积分求解层 —— 与不定积分最大的不同,在于**真值的来源变了**。

不定积分靠"求导回验";定积分没有导数可求,但它的结果就是一个数,
数值求积能给出另一个数 —— 这是比逐点恒等式更硬的证据。
所以这里的验证门是 `quadrature.evaluate`。

策略按"先便宜后昂贵"排:

    0. 数值预检    先算一遍数值,同时**判定敛散性**。
                   如果数值说发散,后面就不用白费力气了 ——
                   而且这一步能拦住一件很危险的事:SymPy 有时会给发散的
                   积分返回一个看起来很正常的有限值。
    1. 直接积分    sp.integrate(f, (x, a, b))
    2. 牛顿-莱布尼茨 用**已验证的**原函数,在上下限取极限。
                   关键:原函数在区间内可能不连续(比如 ln(x−1) 在 x=1),
                   这时必须按不连续点把区间切开,否则 F(b)−F(a) 是错的。
    3. 对称性      对称区间上的奇偶性
    4. 换元换限    用方法卡片给出的代换,上下限一起换
    5. 参数求导    费曼技巧(见 parametric.py)
    6. 级数展开    逐项积分

结果带**成立条件**:带参数的积分往往只在参数的某个范围内成立
(∫_0^∞ x^{s−1}e^{−x}dx = Γ(s) 要求 s>0),这个必须报出来。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import sympy as sp

from . import cas
from .quadrature import NumericResult, close, evaluate
from .solve import solve
from .timeout import ResultSink, remaining_budget, run_with_timeout

# 相对误差容限。数值层在正常题目上能给到 15~20 位有效数字,1e-8 很宽松。
_TOL = 1e-8

# 单条策略的超时,以及整个符号求解的总预算。
# 必须设这两道闸:SymPy 的 integrate 会在一部分积分上无限期地卡住
# (实测 ∫_0^1 ln(1+x)/(1+x²)dx),没有闸的话整个系统会被一道题拖死。
STRATEGY_TIMEOUT = 12.0
TOTAL_BUDGET = 60.0

# 数值层已经判定**发散**之后,符号搜索降级为"第二意见",另给一个小窗口。
# 理由是:发散本身就是答案(题目要的是判敛散),符号路线此时只是在试图
# 推翻数值结论 —— 值得试,但不能和"还没有答案"时一样铺开。
# 实测 ∫_1^∞ x^{-1/2}dx 单独跑 2.6 秒却要 11 秒才返回,差的这部分
# 就是"发散已经定了还照跑全套策略"。
DIVERGENT_STRATEGY_TIMEOUT = 3.0
DIVERGENT_TOTAL = 6.0

# `guarded()` 之外的裸调用,现在统一补上闸门(见 _evaluate_bounded /
# solve_definite 里的 bounded())。
#
# 这两处原来是**完全没有上限**的 —— 那才是真正的缺陷:实测某道含参例题
# 让其中一处陷进去,单核跑十几分钟不返回(README 6d.6)。
#
# 为什么上限给得比较宽松(20 秒):`evaluate` 正常情况下是亚秒级,
# 但含参题要在好几组参数取值上各求一次积,合法地慢到十几秒是常事。
# 卡得太紧会把"慢"变成"错"—— 实测验证上限收到 15 秒时,
# ∫_0^∞ e^{-tx²}dx 在累积负载下会被判成失败,而它单独跑 7 秒就能验证通过。
EVALUATE_TIMEOUT = 20.0      # 单次数值求积
SUBSTITUTION_TIMEOUT = 10.0  # 子问题构造(要 sp.solve 反解新上下限)

# 「直接积分」在第一拍只当**探针**用,给一个短上限。
#
# 理由来自实测:`直接积分` 在 56 道题上累计花掉 35 秒,其中 **12 秒是 d31 一道题**
# 的整段超时 —— 它在根问题上积不出来,却先白等满 12 秒,子问题(真正能解的那条
# 路)才轮到。而**除 d31 外没有一道题的"直接积分"超过 1 秒**。
# 所以"先给 4 秒探一下,不行就先走结构化路线,最后再用足预算回来补一次"
# 几乎不要成本,却能省下那道 12 秒的等待。
DIRECT_PROBE_TIMEOUT = 4.0

# 「区间再现公式」单独的时间上限,比其它策略短得多。
# 它的成败极其两极:能成的时候是零点几秒(两阶段的候选按 count_ops 排过,
# 试前 6 个就见分晓),不能成的时候就是一整段超时白烧。
# 实测 ∫_0^{π/4} ln(1+tan t)dt 上它会烧满 12 秒,而紧跟着的
# 牛顿-莱布尼茨也要 12 秒 —— 一道题先白等 24 秒。
# 给它 4 秒:要么早就出来了,要么基本可以断定出不来。
KING_TIMEOUT = 4.0

# 子问题搜索的上限:最多处理几个问题、换元最多套几层。
# 加上限是为了防止换元链无限延伸(每次换元都产生新问题)。
MAX_PROBLEMS = 8
MAX_DEPTH = 2


@dataclass
class DefiniteAttempt:
    strategy: str
    value: sp.Expr | None
    verify: NumericResult | None = None
    note: str = ""


@dataclass
class DefiniteSolution:
    ok: bool
    x: sp.Symbol
    f: sp.Expr
    lower: sp.Expr
    upper: sp.Expr
    value: sp.Expr | None = None
    value_latex: str = ""
    strategy: str = ""
    proof: str = ""
    conditions: list[str] = field(default_factory=list)
    numeric: NumericResult | None = None
    diverges: bool = False
    divergence_reason: str = ""
    attempts: list[DefiniteAttempt] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    # 「这道题是怎么解出来的」—— 人能读的步骤。
    # 数据一直都在(`_Problem.sub` 记录每次换元、`_Problem.parent` 串成链),
    # 但过去只渲染成一行"换元链:… → 区间再现公式",外加一段原始尝试记录。
    steps: list[str] = field(default_factory=list)


# ================================================================ 参数
def parameters_of(f: sp.Expr, x: sp.Symbol) -> list[sp.Symbol]:
    """被积函数里除积分变量以外的自由符号(就是"参数")。"""
    return sorted((s for s in f.free_symbols if s is not x and s.is_real),
                  key=lambda s: s.name)


def _parameter_combos(params: list[sp.Symbol], count: int = 3,
                      constraints: list | None = None) -> list[dict]:
    """给参数造几组取值。

    **一律取正值。** 这一条是踩过坑之后定下来的:教材里的参数(a、b、n、s、t)
    几乎都是正的,而我们的符号表里 `t`、`s` 只声明了 `real=True`。
    如果按"没有正假设就取一正一负"来采样,∫_0^1 x^t dx 会在 t=−1.5 处发散,
    于是一个完全正确的闭式 1/(t+1) 会被判成"未通过验证"。
    真正需要报告的"参数范围"由 _probe_conditions 单独去探。

    constraints 是 SymPy 给的分支条件(如 Piecewise 里的 `s < 1`)。
    **必须尊重它**:∫_0^∞ x^{s−1}/(1+x)dx = π/sin(πs) 只在 0<s<1 成立,
    如果还拿 s=2、5 去采样,那儿积分本来就发散,一个正确的闭式会被误杀
    (实测就是这样误判的)。有了条件就只在条件内部采样。
    """
    if constraints:
        return _constrained_combos(params, constraints)

    pools = []
    for sym in params:
        if sym.is_integer and sym.is_positive:
            # 取整数,而且**偶数奇数都要有**:Wallis 公式对 n 的奇偶给出不同形式,
            # 只取一种就发现不了"这个闭式只对偶数成立"这类问题。
            pools.append([2, 3, 4])
        else:
            pools.append([0.5, 2.0, 5.0])
    combos = []
    for index in range(count):
        combos.append({sym: pools[i][(index + i) % len(pools[i])]
                       for i, sym in enumerate(params)})
    return combos


# 采样候选池:覆盖 0~1、1 附近、以及负数,方便按条件筛
_CANDIDATE_POOL = (0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 0.1, 5.0, -0.5, -1.5, -0.25)


def _satisfies(sym: sp.Symbol, value, constraints: list) -> bool:
    """把候选值代进约束,看是否满足。判不了就当不满足(保守)。"""
    for condition in constraints:
        try:
            if condition is sp.true:
                continue
            if sym not in condition.free_symbols:
                continue
            substituted = condition.subs(sym, sp.Float(str(value), 25))
            if substituted is sp.true:
                continue
            if substituted is sp.false:
                return False
            try:
                if bool(substituted):
                    continue
            except Exception:
                return False
            # 化简一下再试(如 0<s<1 代 s=0.5 会得到 0<0.5 且 0.5<1)
            try:
                if bool(sp.simplify(substituted)):
                    continue
            except Exception:
                return False
        except Exception:
            return False
    return True


def _constrained_combos(params: list[sp.Symbol], constraints: list,
                        count: int = 3) -> list[dict]:
    """在条件允许的范围内采样。

    每个参数先各自挑出满足约束的候选值,再按"错位搭配"组合起来,
    免得三个参数永远取同一组数(那样会掩盖某些方向的问题)。
    """
    pools = []
    for sym in params:
        usable = [v for v in _CANDIDATE_POOL if _satisfies(sym, v, constraints)]
        if len(usable) < 2:
            # 条件太刁钻或者判不了,退回默认正值池,但只挑能过条件的
            usable = [v for v in (0.5, 2.0, 5.0) if _satisfies(sym, v, constraints)]
        if not usable:
            usable = [0.5, 2.0, 5.0]
        pools.append(usable)
    combos = []
    for index in range(count):
        combos.append({sym: pools[i][(index + i) % len(pools[i])]
                       for i, sym in enumerate(params)})
    return combos


# 用来探"参数取到什么值时积分不成立"的敌意取值
_HOSTILE_VALUES = (-1.5, -0.5, 0)


def _evaluate_bounded(f: sp.Expr, x: sp.Symbol, lo, hi, params: dict | None = None,
                      timeout: float = EVALUATE_TIMEOUT):
    """带硬超时的数值求积。超时返回 None。

    ★ 这是补上的真实缺口:原来 `evaluate` 是**裸调**的,不在任何超时保护之内。
    `guarded()` 只包住"策略",而数值求积在它外面 —— 一旦陷进去就没有墙。

    **超时 ≠ 不符。** 调用方必须把 None 当成"这一组没能求出来"(未知),
    而不是"数值和符号对不上"(矛盾)。这两件事混为一谈的代价很大:
    实测把验证上限收紧到 15 秒时,`∫_0^∞ e^{-tx²}dx` 在累积负载下
    就因为一组求积没算完而被判成失败 —— 一次"慢"直接变成了"错"。
    """
    status, payload = run_with_timeout(
        lambda: evaluate(f, x, lo, hi, params=params), timeout)
    return payload if status == "ok" else None


def _probe_conditions(f: sp.Expr, x: sp.Symbol, lo, hi,
                      params: list[sp.Symbol]) -> list[str]:
    """主动去探参数的有效范围。

    数值上表现为:某个参数取值下积分**发散**,而符号闭式在那里是有限的
    —— 这正是 Piecewise 条件想表达的东西。把它作为"成立条件"报出来,
    比只给一个光秃秃的闭式诚实得多:
        ∫_0^∞ x^{s−1}e^{−x}dx = Γ(s) 只在 s>0 成立,条件不是废话,是答案的一部分。
    """
    notes: list[str] = []
    for sym in params:
        for value in _HOSTILE_VALUES:
            numeric = _evaluate_bounded(f, x, lo, hi, params={sym: value})
            if numeric is None:
                continue
            if numeric.divergent:
                notes.append(f"{sym} 必须使积分收敛({sym}={value} 时发散)")
                break
    return notes


def _verify_value(value: sp.Expr, f: sp.Expr, x: sp.Symbol,
                  lo: sp.Expr, hi: sp.Expr,
                  constraints: list | None = None) -> tuple[bool, str, NumericResult | None]:
    """把符号结果拿到若干组参数取值上做数值核对。

    constraints 是符号层给出的分支条件(如 `0<s<1`),采样会限制在里面 ——
    否则会拿条件之外的参数值去验证,那儿积分本来就发散。
    """
    params = parameters_of(f, x)
    # 结果里带参数但被积函数里没有(比如由积分产生的常数)也算进来
    params = sorted(set(params) | {s for s in value.free_symbols
                                   if s is not x and s.is_real}, key=lambda s: s.name)

    if not params:
        numeric = _evaluate_bounded(f, x, lo, hi)
        if numeric is None:
            return False, "数值层求积超时,无法确认", None
        if not numeric.ok:
            return False, f"数值层无法确认:{numeric.verdict}", numeric
        try:
            claimed = complex(sp.N(value, 30))
        except Exception as exc:
            return False, f"符号结果无法数值化:{type(exc).__name__}", numeric
        agree, rel = close(claimed, complex(numeric.value))
        if agree:
            return True, f"数值核对通过(相对误差 {rel:.2e},方法 {numeric.method})", numeric
        return False, (f"数值不符:符号值 {claimed:.12g},数值 {complex(numeric.value):.12g},"
                       f"相对误差 {rel:.2e}"), numeric

    # 带参数:在几组取值上分别核对(一律取正值,理由见 _parameter_combos)。
    #
    # 三种结果必须分开记:
    #   confirmed   数值和符号一致      —— 支持这个结果
    #   contradicted 数值和符号不一致   —— **否证**这个结果
    #   unknown     求积超时/无法确认   —— 既不支持也不否证
    # 把 unknown 归进 contradicted 会让"慢"变成"错"(实测踩过)。
    confirmed = 0
    contradicted: list[str] = []
    unknown: list[str] = []
    last_numeric = None
    for combo in _parameter_combos(params, constraints=constraints):
        numeric = _evaluate_bounded(f, x, lo, hi, params=combo)
        if numeric is None:
            unknown.append(f"{_fmt_combo(combo)} 处数值求积超时")
            continue
        last_numeric = numeric
        if not numeric.ok:
            unknown.append(f"{_fmt_combo(combo)} 处数值无法确认")
            continue
        try:
            claimed = complex(sp.N(value.subs({k: sp.Float(str(v), 30)
                                               for k, v in combo.items()}), 30))
        except Exception:
            unknown.append(f"{_fmt_combo(combo)} 处符号结果无法数值化")
            continue
        agree, rel = close(claimed, complex(numeric.value))
        if not agree:
            contradicted.append(f"{_fmt_combo(combo)} 处不符(相对误差 {rel:.2e})")

        else:
            confirmed += 1
    # 有一处**明确不符**就否证 —— 这是不可退让的门槛
    if contradicted:
        return (False,
                f"数值否证:{';'.join(contradicted[:3])}", last_numeric)
    # 通过的门槛仍然是"至少两组独立确认",没有因为超时而放松
    if confirmed >= 2:
        extra = f"(另有 {len(unknown)} 组未能求值)" if unknown else ""
        return True, f"在 {confirmed} 组参数取值上数值核对通过{extra}", last_numeric
    reason = ";".join(unknown[:3]) or "无有效核对"
    return False, f"参数取值上数值核对未通过:{reason}", last_numeric


def _fmt_combo(combo: dict) -> str:
    return "(" + ", ".join(f"{k}={v}" for k, v in combo.items()) + ")"


# ================================================================ 条件提取
def _extract_conditions(raw: sp.Expr, value: sp.Expr) -> list[str]:
    """从 SymPy 返回的 Piecewise 里把成立条件提出来。

    带参数的定积分,SymPy 经常返回
        Piecewise((gamma(s), s > 0), (Integral(...), True))
    那个 `s > 0` 就是答案的一部分,不能丢 —— 它决定了这个闭式在哪些参数上成立。
    """
    conditions: list[str] = []
    if isinstance(raw, sp.Piecewise):
        for branch, cond in raw.args:
            if cond is sp.true:
                continue
            text = str(cond)
            if text not in conditions:
                conditions.append(text)
    for symbol in sorted(value.free_symbols, key=lambda s: s.name):
        for assumption, label in ((symbol.is_positive, ">0"),):
            _ = assumption, label
    return conditions


def _condition_assumptions(value: sp.Expr) -> list[str]:
    """结果里用到、但没写进 Piecewise 的隐含假设(如 Γ(s) 要求 s>0)。"""
    notes: list[str] = []
    for node in sp.preorder_traversal(value):
        if isinstance(node, sp.gamma) and node.args:
            arg = node.args[0]
            notes.append(f"Γ({arg}) 要求 {arg} > 0")
        elif isinstance(node, sp.beta):
            notes.append(f"B({', '.join(str(a) for a in node.args)}) 要求参数为正")
        elif isinstance(node, sp.csc) or isinstance(node, sp.sec):
            notes.append("结果含 csc/sec,参数需避开使分母为零的取值")
    deduped: list[str] = []
    for item in notes:
        if item not in deduped:
            deduped.append(item)
    return deduped


# ================================================================ 策略
def _branches_of(condition) -> list:
    """把 And 形式的条件拆成若干条。"""
    if condition is sp.true:
        return []
    if isinstance(condition, sp.And):
        out: list = []
        for arg in condition.args:
            out.extend(_branches_of(arg))
        return out
    return [condition]


def _unwrap_for_parameters(raw: sp.Expr, f: sp.Expr, x: sp.Symbol,
                           lo: sp.Expr, hi: sp.Expr):
    """如果 SymPy 返回 Piecewise,挑出"当前参数下成立"的那个分支。

    返回 **(值, 条件列表)**。条件必须带出来:它是答案的一部分,
    而且决定了后面验证时参数该在什么范围里取值 ——
    ∫_0^∞ x^{s−1}/(1+x)dx = π/sin(πs) 只对 0<s<1 成立,
    不知道这个条件就会拿 s=2 去验证,那儿积分本来就发散。
    """
    if not isinstance(raw, sp.Piecewise):
        return raw, []
    for branch, cond in raw.args:
        if cond is sp.true:
            continue
        if branch.has(sp.Integral):
            continue
        return branch, _branches_of(cond)
    for branch, cond in raw.args:
        if cond is sp.true and not branch.has(sp.Integral):
            return branch, []
    return None, []


def _positive_swapped(f: sp.Expr, x: sp.Symbol):
    """把参数临时换成"正数符号"。

    解析器给 a、b、n、s、t 只声明了 real=True(因为有时代换变量也叫 t,
    加正性假设会误伤)。但对定积分来说,题目里的参数在教材语境下几乎总是正的。
    于是 SymPy 判不了 ∫_0^∞ e^{−tx}dx 在无穷远端的收敛性,直接放弃 ——
    明明加上 t>0 立刻就出结果。

    这里临时换成正性符号重试一次,拿到结果后再换回原符号。
    换回这一步必须做对,否则答案里会出现 a_pos 这种莫名其妙的符号。
    """
    params = parameters_of(f, x)
    if not params:
        return None
    mapping = {}
    for param in params:
        if param.is_positive:
            continue
        positive = sp.Symbol(param.name + "_pos", positive=True)
        mapping[param] = positive
    if not mapping:
        return None
    return f.subs(mapping), mapping


def _restore_parameters(value, mapping: dict):
    """把临时换成正性符号的参数换回原符号。"""
    inverse = {v: k for k, v in mapping.items()}
    try:
        return value.subs(inverse)
    except Exception:
        return value


def _strategy_direct(f, x, lo, hi):
    """策略 1:直接交给 SymPy 求定积分。

    两条细节很关键:

    ① **参数优先按正数处理。** 解析器只给 a、b、s、t 声明了 real=True,
       于是 SymPy 判不了无穷远端的收敛性、直接放弃。而教材语境里这些参数
       几乎总是正的。把参数临时换成正性符号重试,换回后再把"参数需要满足的
       范围"作为成立条件单独报出来 —— 比硬加上假设然后闭口不提要诚实。

    ② **Piecewise 要先拆分支再看有没有 Integral。** SymPy 经常返回
       `Piecewise((π/sin(πs), s<1), (Integral(...), True))` ——
       整个表达式确实"含 Integral",但第一个分支是完全可用的。
       先做 `raw.has(Integral)` 判断会把这种答案白白丢掉。
    """
    attempts = []
    swapped = _positive_swapped(f, x)
    if swapped is not None:
        attempts.append(("(参数按正数处理)", swapped[0]))
    attempts.append(("", f))

    last_note = "SymPy 未能求出(结果仍是 Integral)"
    for label, target in attempts:
        # 走 cas 闸口,而不是裸调 `sp.integrate`。裸调时完全靠外层
        # `guarded()` 的 12 秒兜底,超时之后那个线程变成僵尸继续抢 CPU ——
        # 同一道题在 50 题评估里比单独跑慢 2.4 倍就是这么来的。
        # cas 还保证同一个积分不会被这么多次策略反复重试。
        budget = remaining_budget(STRATEGY_TIMEOUT)
        if budget <= 0:
            last_note = "父层预算已耗尽"
            break
        status, raw = cas.integrate_definite(target, x, lo, hi, budget)
        if status == "throttled":
            last_note = cas.note("throttled")
            continue
        if status == "budget":
            last_note = "父层预算已耗尽"
            break
        if status == "timeout":
            last_note = (f"integrate 超时(>{budget:.0f} 秒,已放弃;"
                         "同一调用不再重复发起)")
            continue
        if status == "error":
            last_note = f"integrate 抛异常:{type(raw).__name__}"
            continue
        if raw is None:
            last_note = "返回 None"
            continue
        if raw.has(sp.AccumBounds):
            return None, raw, "结果是 AccumBounds,说明积分振荡且无极限(不收敛)", []

        # 先试着从 Piecewise 里挑出一个可用分支(连同它的条件)
        value, conditions = _unwrap_for_parameters(raw, target, x, lo, hi)
        if value is None:
            last_note = ("SymPy 未能求出(结果仍是 Integral)"
                         if raw.has(sp.Integral) else "只给出未求值的分支")
            continue
        if value.has(sp.Integral):
            last_note = "挑出的分支里仍含 Integral"
            continue
        if swapped is not None:
            value = _restore_parameters(value, swapped[1])
            # 条件里也可能出现 s_pos,必须一起换回 s ——
            # 否则后面按条件采样时,`s not in (s_pos < 1).free_symbols` 会成立,
            # 约束被静默忽略,又拿条件之外的参数值去验证了。
            conditions = [_restore_parameters(c, swapped[1]) for c in conditions]
        return value, raw, label, conditions
    return None, None, last_note, []


def _discontinuities_in(F: sp.Expr, x: sp.Symbol, lo, hi) -> list[sp.Expr]:
    """原函数在 (lo, hi) 内部的不连续点。"""
    found: set = set()
    try:
        for point in sp.singularities(F, x):
            found.add(point)
    except Exception:
        pass
    try:
        denominator = sp.denom(sp.together(F))
        if denominator != 1:
            for root in sp.solve(sp.Eq(denominator, 0), x):
                found.add(root)
    except Exception:
        pass

    inside = []
    for point in found:
        if not point.is_real:
            continue
        if point in (sp.oo, -sp.oo):
            continue
        # 用**符号比较**判断是否落在区间内部,不要用 lo.is_number 把
        # 带参数的上下限排除掉 —— 那样 ∫_a^{2a} 这类区间的切点就完全不筛了,
        # 区间外的分支点(比如 √(x²−a²) 的 x=−a)会被当成切点,
        # 于是切出一堆退化区间,结果全错。
        # 判不了就**不切开**(保守),总比切错好。
        try:
            if not bool(sp.sympify(lo) < point < sp.sympify(hi)):
                continue
        except Exception:
            continue
        inside.append(point)
    # 排序键要对符号切点容错(切点里可能带参数 a,float 会直接抛异常)
    def _key(node):
        try:
            return float(sp.N(node, 20))
        except Exception:
            return 0.0

    return sorted(set(inside), key=_key)


def _limit_at(F: sp.Expr, x: sp.Symbol, point, side: str):
    try:
        if point in (sp.oo, -sp.oo):
            return sp.limit(F, x, point)
        return sp.limit(F, x, point, side)
    except Exception:
        return None


def _newton_leibniz_from(F: sp.Expr, x: sp.Symbol, lo, hi):
    """给定一个原函数 F,算 F(b)−F(a)。返回 (值, 说明)。

    这一步是**廉价且确定**的(求极限而已),所以特意从"找原函数"里拆出来 ——
    找原函数那一步又贵又不可控,它被超时管着;而这一步可以放在超时外面跑。
    这样即使找 F 的过程超时了,只要 F 已经被抢出来,结果照样能拿到。
    """
    breaks = _discontinuities_in(F, x, lo, hi)
    bounds = [lo] + breaks + [hi]
    total = sp.Integer(0)
    for left, right in zip(bounds, bounds[1:]):
        upper_limit = _limit_at(F, x, right, "-")
        lower_limit = _limit_at(F, x, left, "+")
        if upper_limit is None or lower_limit is None:
            return None, f"在 [{left},{right}] 上求极限失败"
        piece = sp.simplify(upper_limit - lower_limit)
        if piece.has(sp.zoo, sp.nan) or piece in (sp.oo, -sp.oo):
            return None, f"在 [{left},{right}] 上极限发散"
        total += piece
    note = f"原函数 {F}"
    if breaks:
        note += f";原函数在 {breaks} 处不连续,已分段求极限"
    return sp.simplify(total), note


def _strategy_newton_leibniz(f, x, lo, hi, sink=None):
    """策略 2:牛顿-莱布尼茨 —— 用已验证的原函数,在上下限取极限。

    区间内有原函数的不连续点时必须切开:比如 ∫_0^2 dx/(x−1)² 这类,
    直接用 F(2)−F(0) 会给出 0 这种荒谬的结果。

    和 _strategy_direct 一样,参数默认只有 real 假设,SymPy 常常判不了
    上下限处的极限;这里也补一次"把参数视为正数"的重试。

    sink:把这个中转站一路传给 `solve()`,让内层每找到一个原函数就立刻交出来。
    调用方在外面套了超时,**超时之后仍然可以从 sink 里把 F 取出来**,
    再调用 _newton_leibniz_from 把结果算完 —— 这就是"内层成果被外层超时
    掐掉"那个问题的解药。
    """
    attempts = []
    swapped = _positive_swapped(f, x)
    if swapped is not None:
        attempts.append(("(参数按正数处理)", swapped[0]))
    attempts.append(("", f))

    last_note = "没能求出通过验证的原函数"
    for label, target in attempts:
        solution = solve(target, x, sink=sink)
        found = solution.result if solution.ok else None
        if found is None and sink is not None:
            # solve 被超时打断了,但内层可能已经抢出结果了
            harvested = [a for a in sink.peek()]
            if harvested:
                found = harvested[-1].result
        if found is None:
            continue
        value, note = _newton_leibniz_from(found, x, lo, hi)
        if value is None:
            last_note = note
            continue
        if swapped is not None:
            value = _restore_parameters(value, swapped[1])
        return sp.simplify(value), found, f"{label}{note}"
    return None, None, last_note


def _strategy_symmetry(f, x, lo, hi):
    """策略 3:对称区间上的奇偶性。"""
    try:
        if sp.simplify(lo + hi) != 0:
            return None, None, "区间不对称"
    except Exception:
        return None, None, "区间对称性无法判定"
    try:
        negated = sp.simplify(f.subs(x, -x))
    except Exception:
        return None, None, "无法计算 f(−x)"
    if sp.simplify(negated + f) == 0:
        return sp.Integer(0), "odd", "奇函数在对称区间上积分为 0"
    if sp.simplify(negated - f) == 0:
        half = sp.integrate(f, (x, 0, hi))
        if half.has(sp.Integral):
            return None, None, "是偶函数,但半区间积分没算出来"
        return sp.simplify(2 * half), "even", "偶函数,翻倍到半区间"
    return None, None, "既不是奇函数也不是偶函数"


def _simplify_log_arguments(expr: sp.Expr) -> sp.Expr:
    """把每个 log 的**参数**单独化简一遍。

    这一步是 ∫_0^{π/4} ln(1+tan t)dt 的关键。
    反射后两项的和经 logcombine 变成

        log((tan t + 1)·(cot(t + π/4) + 1))

    外层 sp.simplify 完全动不了里面的乘积,但那个乘积其实恒等于 2 ——
    把参数单独拿出来化,整个式子立刻塌成 log 2,积分瞬间出结果。
    只盯着最外层化简是不够的,要深入到对数里面去。
    """
    try:
        return expr.replace(sp.log, lambda arg: sp.log(sp.simplify(arg)))
    except Exception:
        return expr


def _strategy_king_property(f, x, lo, hi):
    """策略 3.5:区间再现公式。

        ∫_a^b f(x)dx = ∫_a^b f(a+b−x)dx

    两式相加再除以 2,得到

        I = (1/2)∫_a^b [f(x) + f(a+b−x)] dx

    右边这个"和"往往比 f 本身简单得多:

        ∫_0^π x·sin x/(1+cos²x) dx   反射后和变成 π·sin x/(1+cos²x),π 可以提出来
        ∫_0^{π/2} sin x/(sin x+cos x)dx  反射后和恒等于 1
        ∫_0^{π/4} ln(1+tan t)dt      反射后两项的对数**加起来是 ln 2**

    最后那个例子说明:化简这一步不能只靠 sp.simplify。
    `log(1+tan t) + log(1+cot(t+π/4))` 用 simplify 是不动的,
    但先用 logcombine 合并成一个对数,内层就会化到 2,于是得到 ln 2。
    所以这里准备了一个**化简阶梯**,逐个候选去试积分。
    """
    try:
        reflected = f.subs(x, lo + hi - x)
    except Exception:
        return None, None, "反射代换失败"

    raw = f + reflected

    # 候选生成是"两阶段组合":先做各种展开,再做各种合并/化简。
    # 顺序不能反 —— 这一点是实测出来的:
    #   sp.simplify((tan t+1)(cot(t+π/4)+1))               → 化不动,还是乘积
    #   sp.simplify(sp.expand_trig(那个乘积))              → 2 ✓
    # 必须先用 expand_trig 把 cot(t+π/4) 展开成 (cot t−1)/(cot t+1),
    # 内层的 +(1) 才能合并、约掉。所以 bases 里要有 expand_trig 过的版本。
    bases: list[sp.Expr] = [raw]
    for step in (lambda e: sp.expand_trig(e),
                 lambda e: sp.trigsimp(e),
                 lambda e: sp.trigsimp(sp.expand_trig(e)),
                 lambda e: sp.simplify(e)):
        try:
            candidate = step(raw)
        except Exception:
            continue
        if not any(candidate == existing for existing in bases):
            bases.append(candidate)

    finishes = (
        lambda e: e,
        lambda e: sp.logcombine(e, force=True),
        lambda e: _simplify_log_arguments(sp.logcombine(e, force=True)),
    )
    candidates: list[sp.Expr] = []
    for base in bases:
        for finish in finishes:
            try:
                candidate = finish(base)
            except Exception:
                continue
            if candidate.has(sp.zoo, sp.nan):
                continue
            if any(candidate == existing for existing in candidates):
                continue
            candidates.append(candidate)

    # 按"简单程度"排序再逐个试积分。
    # 这一条很关键:sp.integrate 对某些中间形式会挂很久,
    # 而真正能积的那个形式(比如塌成 log 2 的那个)通常最简单。
    # 不排序的话时间会全花在注定失败的候选上 —— 实测这个策略
    # 单独跑一次要 210 秒,而答案其实一秒就能出来。
    candidates.sort(key=lambda e: sp.count_ops(e))

    for averaged in candidates[:6]:
        # 反射后和就是 2f(一点没变) → 这个公式帮不上忙
        try:
            if sp.simplify(averaged - 2 * f) == 0:
                continue
        except Exception:
            pass
        try:
            half = sp.integrate(averaged, (x, lo, hi))
        except Exception:
            continue
        if half.has(sp.Integral):
            continue
        return (sp.simplify(half / 2), None,
                f"区间再现公式:I = ½∫[f(x)+f(a+b−x)]dx(化简后为 {averaged})")
    return None, None, "区间再现公式没能化简出可积的形式"


def _strategy_substitution(f, x, lo, hi, plan_subs, depth: int = 0):
    """(已废弃)换元后在本函数内部递归求解 —— 保留它只为说明为什么不能这么做。

    内层递归跑的是同一套策略,每一层都要花掉外层给的时间,
    于是外层超时会把内层**刚刚算出来的结果**一起掐掉。
    实测 ∫_0^1 ln(1+x)/(1+x²)dx 的「换元 + 区间再现」两步链就死在这里:
    外层报「换元换限」超时,而内层其实已经把 ln(1+tan t) 积出来了。

    现在改成由 solve_definite 用**子问题队列**在顶层调度:
    换元只负责"产生一个新问题",求解在新问题自己的预算里进行,
    算出结果后再把整条链映射回原变量、对**原积分**做验证。
    """
    return None, None, "换元改由顶层子问题队列调度"


def _bose_fermi_form(f: sp.Expr, x: sp.Symbol):
    """识别 Bose/Fermi 型被积函数 x^p / (e^{ax} ± 1)。

    返回 (p, a, sign) 或 None:sign = −1 对应 e^{ax}−1,+1 对应 e^{ax}+1。
    """
    factors = f.as_ordered_factors() if isinstance(f, sp.Mul) else [f]
    power = None
    denominator = None
    for factor in factors:
        if isinstance(factor, sp.Pow) and factor.exp == -1:
            denominator = factor.base
            continue
        if factor == x:
            power = sp.Integer(1)
            continue
        if isinstance(factor, sp.Pow) and factor.base == x and factor.exp.is_number:
            power = factor.exp
            continue
        return None
    if power is None or denominator is None:
        return None

    terms = sp.Add.make_args(denominator)
    if len(terms) != 2:
        return None
    exp_term = None
    constant = None
    for term in terms:
        if term.has(sp.exp) or (isinstance(term, sp.Pow) and term.base is sp.E):
            exp_term = term
        elif term.is_number:
            constant = term
    if exp_term is None or constant is None:
        return None
    if constant not in (sp.Integer(1), sp.Integer(-1)):
        return None

    arg = exp_term.args[0] if isinstance(exp_term, sp.exp) else exp_term.exp
    try:
        poly = sp.Poly(sp.expand(arg), x)
    except Exception:
        return None
    if poly.degree() != 1:
        return None
    a = poly.coeff_monomial(x)
    if not a.is_positive:
        return None
    return power, a, int(constant)


def _strategy_bose_fermi(f, x, lo, hi):
    """策略 1.6:Bose/Fermi 型积分(标准结果)。

        ∫_0^∞ x^{s−1}/(e^{x} − 1) dx = Γ(s)·ζ(s)
        ∫_0^∞ x^{s−1}/(e^{x} + 1) dx = (1 − 2^{1−s})·Γ(s)·ζ(s)
    带 a>0 时整体除以 a^s(做一次 ax 的换元即可)。

    这一族 SymPy 是完全放弃的:∫_0^∞ x/(e^x−1)dx = π²/6,它连 meijerg
    都试过仍然返回未求值的 Integral。但它是教科书级的标准结果
    (黑体辐射的 Stefan–Boltzmann 律、Fermi 气体的低温展开都要用),
    所以由标准结果卡片接手 —— 这正是"方法卡片"该发挥作用的地方。
    """
    try:
        if sp.sympify(lo) != 0 or sp.sympify(hi) != sp.oo:
            return None, None, "区间不是 [0, ∞)"
    except Exception:
        return None, None, "区间无法判定"

    matched = _bose_fermi_form(f, x)
    if matched is None:
        return None, None, "不是 x^p/(e^{ax}±1) 形式"

    power, a, constant = matched
    s = sp.simplify(power + 1)          # x^p = x^{s−1} ⇒ s = p+1
    value = sp.gamma(s) * sp.zeta(s) / a**s
    if constant > 0:
        value = (1 - 2 ** (1 - s)) * value
    shape = "e^{ax}−1" if constant < 0 else "e^{ax}+1"
    return (sp.simplify(value), None,
            f"Bose/Fermi 标准结果:∫_0^∞ x^{{s−1}}/({shape})dx,s={s},a={a}")


def _log_sine_form(f: sp.Expr, x: sp.Symbol):
    """把 f 识别成 `k0 + a·ln(sin x) + b·ln(cos x)`(k0 为常数)。

    能识别的写法包括 `ln(sin x cos x)`(= ln sin + ln cos)、
    `ln(tan x)`(= ln sin − ln cos)、`ln(sin²x)`(= 2 ln sin),
    以及它们的任意线性组合。写不出来返回 None。

    用 `expand_log(..., force=True)` 把对数里的乘除拆开。这对符号是**有意**
    放宽的(force 会假设参数为正),但本策略只在 [0, π/2] 与 [0, π] 上生效,
    而在那两个区间内部 sin x > 0、cos x > 0 恒成立,端点只是零测集,
    所以这个放宽是合法的。
    """
    try:
        expanded = sp.expand_log(sp.expand(f), force=True)
    except Exception:
        return None

    k0 = sp.Integer(0)
    a = sp.Integer(0)
    b = sp.Integer(0)
    for term in sp.Add.make_args(expanded):
        if term.has(sp.log):
            coefficient, rest = term.as_coeff_Mul()
            logs = [node for node in sp.preorder_traversal(rest)
                    if isinstance(node, sp.log)]
            if len(logs) != 1:
                return None
            argument = logs[0].args[0]
            if argument == sp.sin(x):
                a += coefficient
            elif argument == sp.cos(x):
                b += coefficient
            else:
                # 只支持**不带缩放**的自变量:ln(sin(kx)) 不做(那要另走换元)
                return None
        else:
            if term.free_symbols:
                return None
            k0 += term
    if a == 0 and b == 0:
        return None
    return k0, a, b


def _strategy_log_sine(f, x, lo, hi):
    """策略 1.7:对数正弦型积分(标准结果)。

        ∫_0^{π/2} ln(sin x) dx = ∫_0^{π/2} ln(cos x) dx = −(π/2)·ln2
        ∫_0^{π}   ln(sin x) dx = −π·ln2

    ## 推导(不是查表,是把教科书那三步写下来)

    记 `I = ∫_0^{π/2} ln(sin x)dx`。

    1. **区间再现** `x → π/2 − x`:sin(π/2−x) = cos x,故 `I = ∫_0^{π/2} ln(cos x)dx`;
    2. 两式相加:`2I = ∫_0^{π/2} ln(sin x cos x)dx`;
    3. 用倍角 `sin x cos x = sin(2x)/2` 并换元 `u = 2x`:
       `∫_0^{π/2} ln(sin 2x)dx = (1/2)∫_0^{π} ln(sin u)du`;
    4. 再用 `sin u` 关于 `u = π/2` 的对称性,`∫_0^{π} ln(sin u)du = 2I`;
       于是 `∫_0^{π/2} ln(sin x cos x)dx = I − (π/2)ln2`;
    5. 代回第 2 步:`2I = I − (π/2)ln2` ⇒ **`I = −(π/2)ln2`**。

    有了 `I`,任意组合 `a·ln(sin x) + b·ln(cos x)` 在 [0, π/2] 上的值就是
    `(a+b)·I`。这一族 SymPy 全线放弃(`integrate`、`meijerg` 都返回未求值的
    Integral),所以和 Bose/Fermi 一样交给标准结果策略接手。

    值最终仍然要过数值验证门 —— 推导写错会当场被否掉。
    """
    try:
        lo, hi = sp.sympify(lo), sp.sympify(hi)
    except Exception:
        return None, None, "区间无法判定"

    matched = _log_sine_form(f, x)
    if matched is None:
        return None, None, "不是 ln(sin x)/ln(cos x) 的线性组合"

    k0, a, b = matched
    base = -sp.pi / 2 * sp.log(2)          # ∫_0^{π/2} ln(sin x)dx

    if lo == 0 and hi == sp.pi / 2:
        value = k0 * sp.pi / 2 + (a + b) * base
        where = "区间再现 x→π/2−x 得 ∫ln(sin)=∫ln(cos),再用倍角 + 半区间对称性"
    elif lo == 0 and hi == sp.pi:
        if b != 0:
            # cos x 在 [0, π] 上变号,对数没有实值意义,不能套
            return None, None, "含 ln(cos x) 且区间为 [0, π],本策略不适用"
        value = k0 * sp.pi + a * 2 * base   # ∫_0^{π} ln(sin x)dx = 2∫_0^{π/2} = −πln2
        where = "先用 sin 关于 π/2 的对称性拆成两个半区间"
    else:
        return None, None, "区间不是 [0, π/2] 或 [0, π]"

    return (sp.simplify(value), None,
            f"对数正弦标准结果:{where};k0={k0}, a={a}, b={b}")


@dataclass
class _Problem:
    """一个待求解的定积分,以及它是怎么从原问题换元变过来的。"""
    f: sp.Expr
    x: sp.Symbol
    lo: sp.Expr
    hi: sp.Expr
    sub: object = None                  # 从父问题到这里所用的代换
    parent: object = None
    depth: int = 0

    def label(self) -> str:
        return f"∫_{{{self.lo}}}^{{{self.hi}}} {self.f} d{self.x}"


def _make_subproblem(problem: _Problem, sub):
    """按代换生成一个新的子问题:只做代换和换限,**不求值**。

    求值交给子问题自己在自己的预算里做 —— 这是上一版最大的教训:
    把递归求解塞进带超时的策略内部,内层成果会被外层超时一起掐掉。
    """
    from .solve import _substitute

    try:
        transformed = _substitute(problem.f, problem.x, sub)
    except Exception:
        return None
    bounds = []
    for point in (problem.lo, problem.hi):
        try:
            roots = sp.solve(sp.Eq(sub.x_expr, point), sub.t)
        except Exception:
            roots = []
        roots = [r for r in roots if r.is_real]
        if not roots:
            return None
        bounds.append(roots[0])
    try:
        if float(sp.N(bounds[0], 20)) > float(sp.N(bounds[1], 20)):
            return None                  # 换限后方向反了,不处理
    except Exception:
        pass
    return _Problem(transformed, sub.t, bounds[0], bounds[1],
                    sub=sub, parent=problem, depth=problem.depth + 1)


def _apply_back(value, sub, target: sp.Symbol):
    """把子问题的解从代换变量换回父问题的变量。"""
    expr = value
    if sub.back_map:
        try:
            expr = sp.expand_trig(expr).subs(dict(sub.back_map))
        except Exception:
            pass
    if sub.t_inverse is not None:
        try:
            expr = expr.subs(sub.t, sub.t_inverse)
        except Exception:
            pass
    else:
        try:
            roots = sp.solve(sp.Eq(sub.x_expr, target), sub.t)
            if roots:
                expr = expr.subs(sub.t, roots[0])
        except Exception:
            return None
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def _map_to_root(value, problem: _Problem, root: _Problem):
    """把子问题的解一路换回根问题的变量。返回 (值, 链条文字) 或 (None, "")。"""
    expr = value
    chain: list[str] = []
    node = problem
    while node is not root:
        if node.sub is None or node.parent is None:
            return None, ""
        expr = _apply_back(expr, node.sub, node.parent.x)
        if expr is None:
            return None, ""
        chain.append(node.sub.name)
        node = node.parent
    return expr, " → ".join(reversed(chain))


def _problem_key(problem: _Problem):
    return (str(problem.f), str(problem.lo), str(problem.hi), str(problem.x))


def _describe_steps(problem: _Problem, root: _Problem, strategy: str,
                    inner_value, mapped, why: str, note: str) -> list[str]:
    """把"这道题是怎么解出来的"渲染成人能读的解题步骤。

    ## 为什么值得单独做

    这条链的信息一直存在:`_Problem.sub` 记录了每一次换元(名字、代换式、
    依据哪张卡片、为什么选它),`_Problem.parent` 把它们串成链。但报告里
    过去只有一行 `采用策略:换元链:x = tan t → 区间再现公式`,
    外加第 ⑦ 节一段**原始的**尝试记录。对一份要给人看的报告来说,
    真正有价值的东西——**这条链本身**——恰好没被呈现。

    这也补上了项目自评里反复出现的那条短板:"没有真正的分步解答"。
    """
    steps: list[str] = [f"原积分 {root.label()}"]

    # root → problem 的换元链(节点存的是"从父问题到自己"的那一步)
    chain: list[_Problem] = []
    node = problem
    while node is not root and node.parent is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()

    for child in chain:
        sub = child.sub
        pieces = [f"换元:{getattr(sub, 'name', '换元')}"]
        x_expr = getattr(sub, "x_expr", None)
        t = getattr(sub, "t", None)
        if x_expr is not None and t is not None:
            pieces.append(f"令 x = {x_expr}(t = {t})")
        if getattr(sub, "card_id", None):
            pieces.append(f"依据卡片 {sub.card_id}")
        if getattr(sub, "why", ""):
            pieces.append(str(sub.why))
        steps.append(" · ".join(pieces) + f" ⟹ 化为 {child.label()}")

    steps.append(f"求解:在{'内层问题' if chain else '原积分'}上用「{strategy}」"
                 f"得到 {inner_value}")
    if chain:
        # ★ 定积分的换元是**等值变换**(上下限同时换),所以内层的值就是原积分的值。
        # 这里**不能**照搬不定积分那套"回代"的说法 —— 那会让人以为还要把结果
        # 换回原变量,而实际上 (比如) ∫_0^1 …dx 换成 ∫_0^{π/4} …dt 之后,
        # 两个积分的**数值完全相同**,不存在第二次代入。
        if str(mapped) != str(inner_value):
            steps.append(f"回代:沿换元链把结果换回原变量 {root.x},得到 {mapped}")
        else:
            steps.append("等值性:换元只是把这个积分改写成另一个**等值**的积分"
                         "(上下限同时变换),内层的值就是原积分的值,无需回代")
    steps.append(f"验证:对**原积分**做高精度数值求积核对 —— {why}")
    if note:
        steps.append(f"说明:{note}")
    return steps


def _divergence_steps(root: _Problem, reason: str) -> list[str]:
    return [
        f"原积分 {root.label()}",
        f"敛散判定:数值层判定发散 —— {reason}",
        "结论:该积分不收敛,没有有限值(这**不是**「没算出来」)",
    ]


def _frullani_limit(base, u: sp.Symbol):
    """求 f 在无穷远的"有效极限"。

    普通极限不存在时退回 **Cesàro 均值** `lim (1/R)∫_0^R f du` —— 这是 Frullani
    公式对 `cos` 这类有界振荡函数仍然成立的原因:
        ∫_0^∞ (cos ax − cos bx)/x dx = ln(b/a)
    这里 f(0) = 1,而 f(∞) 的极限**不存在**(cos 在 ±1 间振荡),
    但它的均值是 0,代进公式正好得到 ln(b/a)。所以不能只用 `sp.limit`
    (`sp.limit(cos(u), u, oo)` 返回的是 AccumBounds,不是数)。
    """
    if base is None:
        return None
    try:
        direct = sp.limit(base, u, sp.oo)
        # ★ 不能只判 `is_finite`:振荡型的极限会返回 `AccumBounds(-1, 1)`,
        #   而 **AccumBounds.is_finite 是 True**(它是一个有界集合)。
        #   只判 is_finite 会把"振荡没有极限"当成"极限是 [-1,1]",
        #   再往下算就得到带 AccumBounds 的表达式,最后在 simplify 里炸掉。
        #   所以必须同时要求它是个**数**且不是 AccumBounds。
        if (direct.is_number and direct.is_finite
                and not isinstance(direct, sp.AccumBounds)):
            return direct
    except Exception:
        pass
    try:
        antiderivative = sp.integrate(base, u)
        if antiderivative is None or antiderivative.has(sp.Integral):
            return None
        mean = (antiderivative - antiderivative.subs(u, 0)) / u
        value = sp.limit(mean, u, sp.oo)
        return value if value.is_finite else None
    except Exception:
        return None


def _strategy_frullani(f, x, lo, hi):
    """策略 1.8:Frullani 积分(标准结果)。

        ∫_0^∞ [f(ax) − f(bx)]/x dx = [f(0) − f(∞)]·ln(b/a)      (a, b > 0)

    ## 为什么补这条策略

    §6d 给这一族建了 `def_standard_frullani` **卡片**(检索能推荐它),
    但**没有对应的求解策略** —— 于是检索说"用 Frullani",求解却只能靠
    费曼技巧绕过去(实测 d49 要 22.5 秒)。卡片与策略成对才算真正接上:
    这和 §6e 补对数正弦是同一件事,只是顺序反了过来。

    判别复用 `_scaled_same_function`(§6i 刚修好的那个特征判别器)——
    它本来只为给检索打标签而写,现在同时给求解用,**一处逻辑两处受益**。
    """
    from .features import _scaled_same_function

    try:
        if sp.sympify(lo) != 0 or sp.sympify(hi) != sp.oo:
            return None, None, "区间不是 [0, ∞)"
    except Exception:
        return None, None, "区间无法判定"

    factors = list(sp.Mul.make_args(f))
    reciprocals = [fa for fa in factors
                   if isinstance(fa, sp.Pow) and fa.base == x and fa.exp == -1]
    if len(reciprocals) != 1:
        return None, None, "不是 …/x 的形状"
    rest = [fa for fa in factors if fa is not reciprocals[0]]
    numerator = sp.expand(sp.Mul(*rest)) if rest else sp.Integer(1)

    matched = _scaled_same_function(numerator, x)
    if matched is None:
        return None, None, "分子不是 c·[f(a·x) − f(b·x)] 的形状"
    head, scale_l, scale_r = matched

    # 两个缩放必须同号且非零,否则 f 在两端的行为不一致,公式不成立
    signs = {getattr(scale_l, "is_positive", None),
             getattr(scale_r, "is_positive", None)}
    if len(signs) != 1 or None in signs:
        return None, None, "两个缩放系数不同号或无法判定正负"
    sign = sp.Integer(1) if signs.pop() else sp.Integer(-1)

    u = sp.Symbol("u", positive=True)
    if head == "recip":
        base = 1 / u
    else:
        try:
            base = head(sign * u)
        except Exception:
            return None, None, "函数头无法重建"

    try:
        at_zero = sp.limit(base.subs(u, x), x, 0, "+")
    except Exception:
        return None, None, "f(0) 取不到"
    if at_zero is None or at_zero.has(sp.zoo, sp.nan, sp.oo):
        return None, None, "f(0) 不是有限值"

    at_inf = _frullani_limit(base, u)
    if at_inf is None or at_inf.has(sp.zoo, sp.nan, sp.oo):
        return None, None, "f(∞) 既没有极限也没有 Cesàro 均值"

    ratio = sp.Abs(scale_r) / sp.Abs(scale_l)
    if ratio == 1:
        return None, None, "两个缩放相同,被积函数恒为 0"
    value = (at_zero - at_inf) * sp.log(ratio)
    # 兜底:万一还是混进了非数(AccumBounds / oo / nan),直接弃权而不是往下传
    if value.has(sp.AccumBounds, sp.zoo, sp.nan, sp.oo):
        return None, None, "结果里混进了非有限量,弃权"
    try:
        value = sp.simplify(value)
    except Exception:
        pass          # simplify 在这类含 Abs 的式子上会炸,那就用未化简的形式

    return (value, None,
            f"Frullani 标准结果:[f(0) − f(∞)]·ln(b/a);"
            f"f(0)={at_zero}, f(∞)={at_inf}, b/a={ratio}")


def _strategy_beta_trig(f, x, lo, hi):
    """策略 1.5:∫_0^{π/2} sinᵐx cosᵏx dx = B((m+1)/2, (k+1)/2) / 2。

    这就是 Wallis 公式的一般形式。SymPy 对 ∫_0^{π/2} sinⁿx dx(指数带符号)
    常常直接放弃,而这条标准结果对 n>−1 一律成立,顺手还能覆盖
    ∫_0^{π/2} sinᵐx cosᵏx dx 这一整族(对应 beta_gamma / wallis_reduction 两张卡片)。
    """
    try:
        if sp.simplify(lo) != 0 or sp.simplify(hi - sp.pi / 2) != 0:
            return None, None, "区间不是 [0, π/2]"
    except Exception:
        return None, None, "区间无法判定"

    factors = f.as_ordered_factors() if isinstance(f, sp.Mul) else [f]
    m = sp.Integer(0)
    k = sp.Integer(0)
    coefficient = sp.Integer(1)
    for factor in factors:
        if factor == sp.sin(x):
            m += 1
        elif factor == sp.cos(x):
            k += 1
        elif isinstance(factor, sp.Pow) and factor.base in (sp.sin(x), sp.cos(x)):
            if factor.base == sp.sin(x):
                m += factor.exp
            else:
                k += factor.exp
        elif factor.is_number:
            coefficient *= factor
        else:
            return None, None, "被积函数不是 sin/cos 的幂次与常数之积"

    if m == 0 and k == 0:
        return None, None, "没有 sin/cos 因子"

    value = coefficient * sp.beta((m + 1) / 2, (k + 1) / 2) / 2
    return (sp.simplify(value), None,
            f"Beta 形式:∫_0^{{π/2}} sin^{m}x cos^{k}x dx = B(({m}+1)/2, ({k}+1)/2)/2")


def _strategy_series(f, x, lo, hi):
    """策略 6:在区间端点展开成幂级数,逐项积分。"""
    for raw_center in (lo, hi, sp.Integer(0)):
        center = sp.sympify(raw_center)
        if not center.is_number:
            continue
        for order in (8, 12, 16):
            try:
                series = f.series(x, center, order).removeO()
            except Exception:
                continue
            terms = sp.Add.make_args(sp.expand(series))
            try:
                total = sum(sp.integrate(term, (x, lo, hi)) for term in terms)
            except Exception:
                continue
            if total.has(sp.Integral):
                continue
            return sp.simplify(total), None, f"在 x={center} 处展开到 {order} 阶后逐项积分"
    return None, None, "级数展开不适用"


# ================================================================ 主入口
def solve_definite(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr,
                   plan_subs=None, use_feynman: bool = True,
                   depth: int = 0, do_numeric: bool = True) -> DefiniteSolution:
    """求解 ∫_lo^hi f dx。只返回通过数值验证的结果(或明确的发散结论)。

    ## 为什么要有"子问题队列"

    换元会把一个问题变成另一个问题,而新问题往往需要**再换一种方法**才积得动。
    比如 ∫_0^1 ln(1+x)/(1+x²)dx 需要"x = tan t + 区间再现公式"两步组合。

    最早的写法是让换元策略在自己内部递归调用 solve_definite。这是错的:
    内层跑的是同一套策略、花的是外层给的时间,于是外层超时会把内层
    **刚刚算出来的结果**一起掐掉 —— 实测就是外层报「换元换限」超时,
    而内层其实已经把 ln(1+tan t) 积出来了。

    现在换元只负责**产生**新问题(变换被积函数 + 反解新上下限,都是廉价操作),
    求解交给队列里的新问题在自己的预算里做;算出结果后再沿着换元链
    一路映射回原变量,并对**原积分**做数值验证。
    这样每一步的成果都不会被上层的时间限制连带毁掉。
    """
    solution = DefiniteSolution(ok=False, x=x, f=f, lower=lo, upper=hi)
    started_at = time.monotonic()

    # ---- 策略 0:数值预检(同时定敛散)
    #
    # ★ 这里原来是一句裸的 `evaluate(f, x, lo, hi)` —— **不在任何超时保护之内**。
    # governed() 只保护"策略",而数值预检、结果核对、参数探查、子问题构造
    # 全都在它之外。实测某些含参例题会让其中一处陷进去,单核跑十几分钟不返回,
    # 而且因为没有闸,连"放弃"的机会都没有(见 README 6d.6)。
    numeric = None
    if do_numeric:
        numeric = _evaluate_bounded(f, x, lo, hi)
        if numeric is None:
            solution.messages.append(
                f"「数值预检」超过 {EVALUATE_TIMEOUT:.0f} 秒未返回,已放弃 —— "
                "只是失去数值层面的独立参照,不影响符号路线")
        solution.numeric = numeric
        if numeric is not None and numeric.divergent:
            solution.diverges = True
            solution.divergence_reason = "  ".join(numeric.notes[-2:]) or numeric.method
            solution.messages.append(f"数值层判定发散({numeric.method})")
        elif numeric is not None and numeric.verdict == "inconclusive":
            solution.messages.append(f"数值层无法判定敛散({numeric.method})")

    # ---- 时间闸门:发散已定就缩短符号搜索 ----
    # 数值层是策略 0,结论先于任何符号尝试产生。它一旦判定发散,
    # 符号搜索就从"找答案"降级为"复核",窗口相应收窄。
    if solution.diverges:
        strategy_timeout = DIVERGENT_STRATEGY_TIMEOUT
        total_budget = min(TOTAL_BUDGET, DIVERGENT_TOTAL)
        solution.messages.append(
            f"数值层已判定发散,符号搜索降级为限时复核"
            f"(单策略 {strategy_timeout:.0f} 秒 / 合计 {total_budget:.0f} 秒)")
    else:
        strategy_timeout = STRATEGY_TIMEOUT
        total_budget = TOTAL_BUDGET

    def budget_left() -> bool:
        return time.monotonic() - started_at < total_budget

    def bounded(label: str, fn, timeout: float, fallback, why: str):
        """把 `guarded()` **之外**的调用也放进带超时的守护线程里。

        guarded() 只包住"策略"。而这几处是裸调的:

          * 数值预检 `evaluate`(策略 0)
          * 结果核对 `_verify_value`(它内部要跑好几组参数,每组一次 evaluate)
          * 参数范围探查 `_probe_conditions`(纯增值信息)
          * 子问题构造 `_make_subproblem`(内部要 sp.solve 反解新上下限)

        实测某道含参例题会让其中一处陷进去,单核十几分钟不返回。现在统一
        给它们一个上限:超时就放弃这一处,并**在报告里写明**放弃了什么 ——
        而不是让整道题无声无息地卡死。
        """
        left = total_budget - (time.monotonic() - started_at)
        if left <= 0:
            return fallback
        status, payload = run_with_timeout(fn, min(timeout, left))
        if status != "ok":
            if not any(label in m for m in solution.messages):
                solution.messages.append(f"「{label}」{why}")
            return fallback
        return payload

    def guarded(label: str, fn, timeout: float | None = None):
        """把一条策略放进带超时的守护线程里跑。

        返回 (状态, 值),状态 ∈ {"ok", "timeout", "error", "budget"}。
        """
        if timeout is None:
            timeout = strategy_timeout
        elapsed = time.monotonic() - started_at
        if elapsed > total_budget:
            if not any("总时间预算" in m for m in solution.messages):
                solution.messages.append(
                    f"已达总时间预算 {total_budget:.0f} 秒,后续策略不再尝试")
            return "budget", None
        status, payload = run_with_timeout(fn, min(timeout, total_budget - elapsed))
        if status == "timeout":
            solution.messages.append(f"「{label}」超时(>{timeout:.0f} 秒,已放弃)")
            return "timeout", None
        if status == "budget":
            # 更内层已经发现预算耗尽,连启动都没启动
            solution.messages.append(f"「{label}」因父层预算耗尽而未执行")
            return "budget", None
        if status == "error":
            solution.messages.append(f"「{label}」出错:{type(payload).__name__}: {payload}")
            return "error", None
        return "ok", payload

    def run_strategy(label: str, fn, timeout: float | None = None):
        """带超时地跑一条策略,把结果规整成 (值, 附加, 说明, 条件)。"""
        status, payload = guarded(label, fn, timeout)
        if status != "ok" or payload is None:
            return None, None, "", []
        if isinstance(payload, tuple):
            if len(payload) == 4:
                return payload
            if len(payload) == 3:
                return payload[0], payload[1], payload[2], []
        return payload, None, "", []

    def accept(attempt: DefiniteAttempt, raw, conditions) -> DefiniteSolution:
        solution.ok = True
        solution.value = attempt.value
        solution.strategy = attempt.strategy
        solution.proof = attempt.note
        try:
            solution.value_latex = sp.latex(attempt.value)
        except Exception:
            solution.value_latex = str(attempt.value)
        # conditions 里可能是 SymPy 的 Relational 对象(直接来自 Piecewise),
        # 报告和展示都要字符串,统一转一下。
        texts = [str(c) for c in (conditions or [])]
        if raw is not None and not texts:
            texts = _extract_conditions(raw, attempt.value)
        solution.conditions = texts
        solution.conditions += [c for c in _condition_assumptions(attempt.value)
                                if c not in solution.conditions]
        # 主动探一下参数的有效范围:闭式在哪些参数上成立,是答案的一部分。
        # 逐次求积都有硬超时(见 _evaluate_bounded),所以这里不再另设总闸。
        probe_params = parameters_of(f, x) or [
            s for s in attempt.value.free_symbols if s is not x and s.is_real]
        if probe_params:
            for note in _probe_conditions(f, x, lo, hi,
                                          sorted(probe_params, key=lambda s: s.name)):
                if note not in solution.conditions:
                    solution.conditions.append(note)
        return solution

    # ---- 子问题队列 ----
    # 队列里放的是 (问题, 阶段)。阶段分两拍:
    #   "cheap"     先跑便宜的策略,然后**生成换元子问题**并把它们插到队首,
    #               自己则以 "expensive" 排到队尾;
    #   "expensive" 等子问题都试过之后再回来跑昂贵的策略。
    # 这样安排的原因:如果生成子问题后还继续跑自己的昂贵策略,
    # 子问题会被前面耗掉的时间预算堵死(实测 d31 就是这样,
    # 子问题明明能解,却直到 60 秒预算耗尽都没轮到它)。
    root = _Problem(f, x, lo, hi)
    queue: list[tuple[_Problem, str]] = [(root, "cheap")]
    visited = {_problem_key(root)}
    problems_done = 0

    def consider(problem: _Problem, value, raw, note: str, conditions: list,
                 strategy: str):
        """接受一个候选值:映射回根变量 → 对**根积分**做数值验证。"""
        if value is None:
            return None
        if problem is root:
            mapped, chain = value, ""
        else:
            mapped, chain = _map_to_root(value, problem, root)
            if mapped is None:
                solution.messages.append(
                    f"子问题 {problem.label()} 有解,但换不回原变量")
                return None
        # 结果核对的每一组求积都有自己的硬超时,所以这里**不**再套一层总闸:
        # 套了会把"慢"变成"错"(实测把上限定到 15 秒时 d47 直接判失败)。
        verified, why, numer = _verify_value(mapped, f, x, lo, hi,
                                             constraints=conditions)
        label = f"换元链:{chain} → {strategy}" if chain else strategy
        attempt = DefiniteAttempt(label, mapped, numer, f"{note};{why}")
        solution.attempts.append(attempt)
        if verified:
            done = accept(attempt, raw, conditions)
            done.steps = _describe_steps(problem, root, strategy, value, mapped,
                                         why, note)
            return done
        return None

    while queue and problems_done < MAX_PROBLEMS:
        if not budget_left():
            solution.messages.append("总时间预算用尽,停止搜索")
            break
        problem, stage = queue.pop(0)

        # ==================== 第二拍:昂贵的策略 ====================
        if stage == "expensive":
            # 顺序:先试**便宜**的"区间再现公式",再上最贵的"牛顿-莱布尼茨"。
            # 两条都是要找原函数,但区间再现只是把 f 改写成 f(a+b−x) 再试一次
            # 积分,廉价得多。实测 ∫_0^1 ln(1+x)/(1+x²)dx:换元到 tan 之后,
            # 区间再现 0.2 秒就出结果,而排在它前面的牛顿-莱布尼茨会先烧满
            # 12 秒什么都没给。顺序反过来,这道题省下整整一次超时。
            # (顺序是语义的一部分,不是随便排的 —— 改动必须重跑回归。)
            value, raw, note, conds = run_strategy(
                "区间再现公式",
                lambda p=problem: _strategy_king_property(p.f, p.x, p.lo, p.hi),
                timeout=KING_TIMEOUT)
            if value is not None:
                done = consider(problem, value, None, note, conds, "区间再现公式")
                if done:
                    return done

            # 「牛顿-莱布尼茨」是这条流水线上最贵的一步(要找一个原函数)。
            # 给它配一个成果中转站:即使整条策略超时被放弃,
            # 内层抢出来的原函数我们照样接住,然后在**超时之外**
            # 把 F(b)−F(a) 这步廉价运算补完。
            antiderivative_sink = ResultSink(limit=4)
            status, payload = guarded(
                "牛顿-莱布尼茨",
                lambda p=problem, s=antiderivative_sink: _strategy_newton_leibniz(
                    p.f, p.x, p.lo, p.hi, sink=s))
            harvested_value = None
            if status == "ok" and payload:
                value, raw, note = payload
                if value is not None:
                    done = consider(problem, value, None, note, [],
                                    "牛顿-莱布尼茨(用已验证的原函数)")
                    if done:
                        return done
                elif note:
                    solution.messages.append(f"牛顿-莱布尼茨路线:{note}")
            if status in ("timeout", "budget"):
                # ★ 抢救:内层在超时之前已经找到了原函数
                for attempt in antiderivative_sink.drain():
                    rescued, note = _newton_leibniz_from(
                        attempt.result, problem.x, problem.lo, problem.hi)
                    if rescued is None:
                        continue
                    solution.messages.append(
                        "「牛顿-莱布尼茨」超时,但从成果中转站里抢救出了原函数,"
                        "在超时之外补算了下限上限")
                    done = consider(problem, sp.simplify(rescued), None, note, [],
                                    "牛顿-莱布尼茨(超时后抢救出的原函数)")
                    if done:
                        return done
                    harvested_value = rescued
                if harvested_value is None:
                    solution.messages.append(
                        "「牛顿-莱布尼茨」超时,且中转站里没有可用的原函数")

            if use_feynman and problem is root:
                from .parametric import feynman
                status, results = guarded("费曼技巧", lambda: feynman(f, x, lo, hi))
                for result in (results or []):
                    done = consider(problem, result.value, None, result.evidence, [],
                                    f"费曼技巧(对 {result.param} 求导)")
                    if done:
                        solution.messages.append(result.evidence)
                        return done
                    solution.messages.append(
                        f"费曼技巧对 {result.param} 求导得到 {result.value},但未通过验证")

            value, raw, note, conds = run_strategy(
                "级数展开", lambda p=problem: _strategy_series(p.f, p.x, p.lo, p.hi))
            if value is not None:
                done = consider(problem, value, None, note, conds, "幂级数逐项积分")
                if done:
                    return done

            # ★ 最后再用**足额预算**回来补一次「直接积分」。
            #   第一拍只给了它 DIRECT_PROBE_TIMEOUT(探针),这里才是正式的一试。
            #   走到这里说明结构化路线全都没成,那这 12 秒就不算白花。
            #   `cas` 的记忆化不会挡住这次重试:上一次超时的线程已经被**杀掉**,
            #   它记录的实际耗时(约 4 秒)小于这次的预算(12 秒),
            #   所以 cas 会放行 —— 这正是 §6c 里"超时记忆要能自我解除"那条设计的用处。
            value, raw, note, conds = run_strategy(
                "直接积分(足额重试)",
                lambda p=problem: _strategy_direct(p.f, p.x, p.lo, p.hi))
            if value is not None:
                done = consider(problem, value, raw, note, conds,
                                "直接积分(足额重试)")
                if done:
                    return done
            continue

        # ==================== 第一拍:便宜的策略 ====================
        problems_done += 1
        local = problem is not root
        if local:
            solution.messages.append(f"转去求解换元后的子问题:{problem.label()}")

        # ★ 顺序:先试四个**形状识别型标准结果**,再上通用的「直接积分」。
        #
        # 它们各自只认一种形状,不匹配时几乎零成本地放弃;一旦匹配就是闭式,
        # 而通用的 `sp.integrate` 在这些题上反而很慢 —— 实测
        # `∫_0^∞ (cos ax − cos bx)/x dx` 让 SymPy 烧了 **28 秒**才给结果,
        # 而 Frullani 识别器是毫秒级的。识别器的结果同样要过数值验证门,
        # 认错了会被否掉并继续往下走,所以把顺序提前没有正确性风险。
        value, raw, note, conds = run_strategy(
            "Beta 形式", lambda p=problem: _strategy_beta_trig(p.f, p.x, p.lo, p.hi))
        if value is not None:
            done = consider(problem, value, None, note, conds,
                            "Beta 函数形式(π/2 上的三角函数幂)")
            if done:
                return done

        value, raw, note, conds = run_strategy(
            "Bose/Fermi 标准结果",
            lambda p=problem: _strategy_bose_fermi(p.f, p.x, p.lo, p.hi))
        if value is not None:
            done = consider(problem, value, None, note, conds, "Bose/Fermi 标准结果")
            if done:
                return done

        value, raw, note, conds = run_strategy(
            "对数正弦标准结果",
            lambda p=problem: _strategy_log_sine(p.f, p.x, p.lo, p.hi))
        if value is not None:
            done = consider(problem, value, None, note, conds, "对数正弦标准结果")
            if done:
                return done

        value, raw, note, conds = run_strategy(
            "Frullani 标准结果",
            lambda p=problem: _strategy_frullani(p.f, p.x, p.lo, p.hi))
        if value is not None:
            done = consider(problem, value, None, note, conds, "Frullani 标准结果")
            if done:
                return done

        value, raw, note, conds = run_strategy(
            "直接积分",
            lambda p=problem: _strategy_direct(p.f, p.x, p.lo, p.hi),
            timeout=DIRECT_PROBE_TIMEOUT)
        if value is not None:
            done = consider(problem, value, raw, note, conds,
                            "直接积分(SymPy 内置定积分)")
            if done:
                return done
        elif raw is not None and raw.has(sp.AccumBounds):
            solution.diverges = True
            solution.divergence_reason = "SymPy 返回 AccumBounds:积分振荡且不收敛"
            solution.messages.append("SymPy 判定该积分振荡,无极限")

        value, raw, note, conds = run_strategy(
            "对称性", lambda p=problem: _strategy_symmetry(p.f, p.x, p.lo, p.hi))
        if value is not None:
            done = consider(problem, value, None, note, conds, f"对称性({raw})")
            if done:
                return done

        # ---- 生成换元子问题:插到队首,立刻下潜去试 ----
        # 子问题排在前面,而**自己的昂贵策略紧跟在子问题之后** ——
        # 这样"子问题彻底试完"才轮到自己的昂贵策略。
        # 如果只是简单地 append 到队尾,根问题的昂贵策略会抢在
        # 子问题的昂贵策略之前把预算耗光(实测 d31 就是这样:
        # 子问题已经生成、答案也近在眼前,却始终没轮到)。
        if not local and plan_subs and problem.depth < MAX_DEPTH:
            inserted = 0
            for sub in plan_subs:
                # 子问题构造内部要 sp.solve 反解新上下限 —— 也在超时保护之外。
                child = bounded(
                    "换元换限",
                    lambda s=sub: _make_subproblem(problem, s),
                    SUBSTITUTION_TIMEOUT, None,
                    "超时未返回,改用下一种换元")
                if child is None:
                    continue
                key = _problem_key(child)
                if key in visited:
                    continue
                visited.add(key)
                queue.insert(inserted, (child, "cheap"))
                inserted += 1
                solution.messages.append(f"换元生成子问题:{sub.name} → {child.label()}")
            queue.insert(inserted, (problem, "expensive"))
        else:
            queue.append((problem, "expensive"))

    # ---- 都没成:如果数值说发散,就明确报告发散
    if solution.diverges:
        solution.messages.append("所有符号路线都没能给出有限结果,与数值层的发散判定一致")
        if not solution.steps:
            solution.steps = _divergence_steps(root, solution.divergence_reason
                                               or "数值层判定发散")
    return solution
