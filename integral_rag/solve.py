"""执行层 —— 把「该用什么方法」变成「真的算出来」。

设计要点(这是把 RAG 和 CAS 接起来的地方):

  1. 策略是**一个组合**,不是一招。
     直接积分 → 恒等变形后再积 → 按检索到的方法卡片做代换 → manualintegrate 兜底。
     SymPy 的 integrate 在教科书级题目上大概能吃掉七成,剩下三成需要有人
     告诉它"先做个代换",而"做哪个代换"正是检索层给出的答案。

  2. 每一个候选结果都必须过验证层,而且**一律拿原始被积函数 f 去验证**。
     做代换、做恒等变形都会产生新的表达式,但判定标准只有一个:导数等于 f。

  3. 方法链(步骤)来自 sympy.integrals.manualintegrate.integral_steps,
     它会把解题过程表示成一棵规则树(URule / PartsRule / RewriteRule …)。
     我们把它翻译成中文步骤,作为"为什么这么做"的解释材料。

关于 manualintegrate:它的**返回值**有时比 integrate 更难看(会退回
Piecewise、Abs 等形式),所以只把它当作步骤来源和最后的兜底策略;
它的结果同样要过验证层才敢用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import sympy as sp

from . import cas
from .timeout import ResultSink, budget_exhausted, remaining_budget, run_with_timeout
from .verify import VerifyResult, verify_antiderivative

# ---------------------------------------------------------------- 规则翻译
_RULE_LABEL = {
    "ConstantRule": "常数积分",
    "PowerRule": "幂函数公式",
    "ExpRule": "指数函数公式",
    "SinRule": "正弦公式",
    "CosRule": "余弦公式",
    "TanRule": "正切公式",
    "SecRule": "正割公式",
    "CscRule": "余割公式",
    "CotRule": "余切公式",
    "ReciprocalRule": "∫du/u 公式(得对数)",
    "LogRule": "对数函数公式",
    "ArcsinRule": "反正弦公式",
    "ArctanRule": "反正切公式",
    "SinhRule": "双曲正弦公式",
    "CoshRule": "双曲余弦公式",
    "ConstantTimesRule": "提出常数因子",
    "AddRule": "拆成逐项积分",
    "MulRule": "乘积法则",
    "URule": "换元(凑微分)",
    "PartsRule": "分部积分",
    "CyclicPartsRule": "循环分部积分",
    "RewriteRule": "代数/三角恒等变形",
    "AlternativeRule": "多路尝试",
    "SqrtQuadraticRule": "二次根式标准公式",
    "ReciprocalSqrtQuadraticRule": "二次根式倒数公式",
    "SqrtQuadraticDenomRule": "二次根式作分母",
    "TrigRule": "三角公式",
    "ExpBaseRule": "一般底数指数",
    "NestedPowRule": "嵌套幂",
    "InverseHyperbolicRule": "反双曲函数",
    "DontKnowRule": "未识别出可用的积分规则",
}


def _is_rule(obj) -> bool:
    return type(obj).__module__.startswith("sympy.integrals.manualintegrate")


def _rule_detail(rule) -> str:
    """给一条规则配上可读的细节,比如"令 u = 3x+1"。"""
    name = type(rule).__name__
    try:
        if name == "URule":
            return f"令 u = {rule.u_func}"
        if name == "PartsRule":
            return f"u = {rule.u},dv = {rule.dv} dx"
        if name == "CyclicPartsRule":
            return "两次分部后回到原积分,解方程得结果"
        if name == "RewriteRule":
            return f"改写为 {rule.rewritten}"
        if name == "ConstantTimesRule":
            return f"提出因子 {rule.constant}"
        if name == "SqrtQuadraticRule":
            return f"二次根式参数 a={rule.a}, b={rule.b}, c={rule.c}"
        if name == "PowerRule":
            return f"∫u^{rule.exp} du"
        if name == "ExpRule":
            return f"∫{rule.base}^{rule.exp} d({rule.exp})"
        if name in ("ReciprocalRule",):
            return f"∫d({rule.base})/({rule.base})"
    except Exception:
        pass
    return ""


def step_trace(f: sp.Expr, x: sp.Symbol, max_depth: int = 12) -> tuple[list[dict], bool]:
    """把被积函数的解题过程抽成中文步骤列表。

    返回 (步骤列表, 是否被识别)。步骤列表为 [] 且 identified=False
    表示 SymPy 的规则库不认识这个积分(不代表积不出来)。
    """
    try:
        from sympy.integrals.manualintegrate import integral_steps
    except Exception:
        return [], False

    try:
        root = integral_steps(f, x)
    except Exception:
        return [], False

    if root is None or type(root).__name__ == "DontKnowRule":
        return [], False

    trace: list[dict] = []
    seen: set[int] = set()

    def walk(rule, depth: int) -> None:
        if rule is None or depth > max_depth or id(rule) in seen:
            return
        seen.add(id(rule))
        name = type(rule).__name__
        if name == "DontKnowRule":
            trace.append({"depth": depth, "rule": name,
                          "label": _RULE_LABEL[name], "detail": "该子问题未被识别"})
            return
        trace.append({
            "depth": depth,
            "rule": name,
            "label": _RULE_LABEL.get(name, name),
            "detail": _rule_detail(rule),
            "integrand": str(rule.integrand) if getattr(rule, "integrand", None) is not None else "",
        })

        # 只跟随一条主路径:AlternativeRule 取第一个备选,避免指数级展开
        children: list = []
        for attr in ("substep", "v_step", "second_step"):
            value = getattr(rule, attr, None)
            if _is_rule(value):
                children.append(value)
        for attr in ("substeps", "parts_rules", "alternatives"):
            value = getattr(rule, attr, None)
            if isinstance(value, list):
                if attr == "alternatives" and value:
                    children.append(value[0])
                elif attr != "alternatives":
                    children.extend(v for v in value if _is_rule(v))
        for child in children:
            walk(child, depth + 1)

    walk(root, 0)
    return trace, True


# ---------------------------------------------------------------- 数据结构
@dataclass
class Substitution:
    """一个候选代换 x = x_expr(t)。由规划层给出(依据检索到的方法卡片)。"""

    name: str
    x_expr: sp.Expr
    t: sp.Symbol
    card_id: str | None = None
    why: str = ""
    # 在 x → x_expr(t) 之前先做的替换(按顺序)。
    # 欧拉替换必须用这个:sqrt(ax²+bx+c) 并不是"把 x 换掉就能化简"的,
    # SymPy 盲目代入只会得到一个仍然带根号的式子。必须先显式把根式
    # 换成 t − √a·x,根号才会真正消失。
    pre_subs: list[tuple[sp.Expr, sp.Expr]] = field(default_factory=list)
    # 逆向映射:三角/双曲函数 → 用 x 表达的等价式。回代时先套这个,
    # 得到的就是教材里的标准形式。
    back_map: dict[sp.Expr, sp.Expr] = field(default_factory=dict)
    # t 用 x 表达的直接形式(如 asin(x/a))。
    t_inverse: sp.Expr | None = None


@dataclass
class Attempt:
    strategy: str
    integrand: sp.Expr
    variable: sp.Symbol
    result: sp.Expr | None
    verify: VerifyResult | None
    note: str = ""
    # ★ 走代换路线的两步中间产物。以前**没有**记,于是"代换链"里缺了
    #   最关键的一环:中间原函数是在 t 里求出来的,回代之后才变成 x 的表达式。
    #   报告里因此只能给出一段"代换之后那个积分"的规则库路径,
    #   而看不出它和原积分是怎么接上的。
    intermediate: sp.Expr | None = None      # t 里的原函数
    substitution: object | None = None       # 用的那次 Substitution


@dataclass
class Solution:
    ok: bool
    x: sp.Symbol
    f: sp.Expr
    result: sp.Expr | None = None
    strategy: str = ""
    proof: str = ""
    detail: str = ""
    trace: list[dict] = field(default_factory=list)
    trace_var: sp.Symbol | None = None
    trace_identified: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # 人能读的解题步骤(与定积分那边的 `DefiniteSolution.steps` 对齐)
    steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 单次尝试
INTEGRATE_TIMEOUT = 20.0

# 单个"恒等变形"的时间上限(见 _rewrite_candidates)。
REWRITE_TIMEOUT = 5.0

# 一旦拿到**通过验证**的结果,后续尝试只是在挑更好看的形式。
# 这段"形式择优"是增值工作,必须限时,不能让它拖住已经拿到的答案。
FORM_BUDGET = 8.0


def _try_integrate(expr: sp.Expr, var: sp.Symbol) -> tuple[sp.Expr | None, str]:
    """对 expr 求关于 var 的原函数,返回 (结果, 说明)。

    带超时:SymPy 的 integrate 会在一部分积分上陷进去出不来
    (实测 ∫ln(1+x)/(1+x²)dx),不设闸的话整个系统会被一道题拖死。

    注意这里的 20 秒是**上限**而不是保证:`run_with_timeout` 会把它和父层
    剩余预算取较小值。在定积分那边,整条 `牛顿-莱布尼茨` 策略只给了 12 秒,
    所以这里实际最多只能拿到 12 秒 —— 这是有意的(见 timeout.py 病一)。
    """
    if expr.has(sp.Integral):
        return None, "输入本身含未求值的积分"

    budget = remaining_budget(INTEGRATE_TIMEOUT)
    if budget <= 0:
        return None, "父层预算已耗尽,跳过本次积分"
    # 走 cas 闸口,而不是直接 run_with_timeout:同一个积分(比如
    # `∫e^{-t x²}dx`)会被好几条策略各调一次,每次烧满自己的预算。
    # cas 记住"这个调用超时过",后续策略连发起都不发起。
    status, payload = cas.integrate_indef(expr, var, budget)
    if status == "timeout":
        return None, f"integrate 超时(>{budget:.0f} 秒,已放弃;同一调用不再重复发起)"
    if status == "throttled":
        return None, cas.note("throttled")
    if status == "budget":
        return None, "父层预算已耗尽"
    if status == "error":
        return None, f"integrate 抛异常:{type(payload).__name__}"
    result = payload
    if result is None:
        return None, "integrate 返回 None"
    if result.has(sp.Integral):
        return None, "SymPy 未能积出(结果仍是 Integral)"
    return result, ""


def _verified_attempt(strategy: str, integrand: sp.Expr, variable: sp.Symbol,
                      f_original: sp.Expr, x: sp.Symbol, note: str = "") -> Attempt:
    result, msg = _try_integrate(integrand, variable)
    if result is None:
        return Attempt(strategy, integrand, variable, None, None, note or msg)
    check = verify_antiderivative(result, f_original, x)
    return Attempt(strategy, integrand, variable, result, check, note)


# ---------------------------------------------------------------- 代换
def _drop_abs(expr: sp.Expr) -> sp.Expr:
    """把中段出现的 |u| 直接换成 u。

    这是**有意的激进化简**。三角替换后 √(a²tan²t) 会被 SymPy 写成 a|tan t|,
    而带 Abs 的表达式 integrate 经常直接抛异常或原样返回 Integral。
    去掉 Abs 相当于只在 t 的一个单调区间上求解 —— 只要最终结果能通过
    "求导回验 + 数值交叉验证",这个区间上的答案就是对的;
    验证不过就整条尝试作废。所以这里激进是安全的。
    """
    try:
        return expr.replace(sp.Abs, lambda arg: arg)
    except Exception:
        return expr


def _substitute(f: sp.Expr, x: sp.Symbol, sub: Substitution):
    """按 x = g(t) 把 ∫f(x)dx 变成 ∫f(g(t))·g'(t)dt。

    先做 pre_subs(把根式整体换掉),再做 x → g(t)。顺序不能颠倒。
    """
    expr = f
    for lhs, rhs in sub.pre_subs:
        expr = expr.subs(lhs, rhs)
    g = sub.x_expr
    t = sub.t
    g_prime = sp.diff(g, t)
    transformed = sp.simplify(expr.subs(x, g) * g_prime)
    return _drop_abs(transformed)


def _back_substitute(F_t: sp.Expr, sub: Substitution, x: sp.Symbol) -> list[sp.Expr]:
    """把结果从 t 换回 x,返回若干**不同的写法**供择优。

    两条路线都走,谁的结果更干净由验证层和可读性打分决定:

    路线 A(优先):用预先写好的逆向映射。三角替换的逆向关系是确定的
        (sin t = x/a、cos t = √(a²−x²)/a ……),走这条路得到的就是
        教材里的标准形式。
    路线 B(兜底+交叉验证):让 SymPy 自己解反函数。它有时候解出来是
        -I*log((a + √(a²−x²))/x) 这种复数对数形式,可读性很差;
        但有时候也只有它能解,所以不能省。
    """
    t = sub.t
    variants: list[sp.Expr] = []

    if sub.back_map or sub.t_inverse is not None:
        try:
            value = sp.expand_trig(F_t)     # 先把 sin(2t) 这类多倍角拆开
        except Exception:
            value = F_t
        try:
            if sub.back_map:
                value = value.subs(dict(sub.back_map))
            if sub.t_inverse is not None:
                value = value.subs(t, sub.t_inverse)
            variants.append(value)
        except Exception:
            pass
        for candidate in list(variants):
            try:
                simplified = sp.simplify(candidate)
                if simplified != candidate:
                    variants.append(simplified)
            except Exception:
                pass

    try:
        solutions = sp.solve(sp.Eq(sub.x_expr, x), t)
    except Exception:
        solutions = []
    for sol in solutions:
        try:
            raw = F_t.subs(t, sol)
        except Exception:
            continue
        variants.append(raw)
        try:
            simplified = sp.simplify(raw)
            if simplified != raw:
                variants.append(simplified)
        except Exception:
            pass

    unique: list[sp.Expr] = []
    for expr in variants:
        if not any(expr == u for u in unique):
            unique.append(expr)
    return unique


# ---------------------------------------------------------------- 结果择优
# 验证通过只是底线。SymPy 有时会给出含 Piecewise / Abs / 复数分支的原函数,
# 数学上没错,但完全不是教材里的形式。所以当直接积分的结果明显"脏"的时候,
# 系统会继续尝试方法卡片给出的代换,取更干净的那个。
CLEAN_THRESHOLD = 3.0

# sympy 顶层没有 ImaginaryUnit 这个名字,只能从符号 I 拿它的类型
_IMAGINARY_UNIT = type(sp.I)


def ugliness(expr: sp.Expr) -> float:
    """结果的"可读性代价",越大越不适合直接展示给人看。

    任何一步出问题都返回一个很大的值(当作"很脏"),绝不因为打分失败
    就把整个求解流程带崩。
    """
    try:
        score = 0.0
        score += 8.0 * len(expr.atoms(sp.Piecewise))
        score += 4.0 * len(expr.atoms(sp.Abs))
        for func in (sp.floor, sp.ceiling, sp.sign):
            score += 5.0 * len(expr.atoms(func))
        score += 5.0 * len(expr.atoms(sp.re, sp.im))
        score += 2.0 * len(expr.atoms(_IMAGINARY_UNIT))
        for func in (sp.acosh, sp.asinh, sp.atanh, sp.acoth):
            score += 1.5 * len(expr.atoms(func))
        score += 0.005 * len(str(expr))
        return score
    except Exception:
        return 99.0


# ---------------------------------------------------------------- 组合策略
def _rewrite_candidates(f: sp.Expr, x: sp.Symbol):
    """对 f 做保持恒等的变形,给 SymPy 换一个更容易积的形式。

    ## 两个性能要点,都是实测踩出来的

    ① **必须是生成器,不能先算完再返回。**
    里面的 `sp.simplify` / `sp.factor` / `sp.apart` 在三角积上是灾难:
    实测 `∫sin 3x·cos 5x dx` 上整套变形要 **80 秒以上**,而调用方一旦
    拿到通过验证的结果就会提前收工 —— 那些变形根本不需要算。
    先算完再返回,等于把 80 秒无条件付掉。

    ② **每个变形单独套超时。**
    这一步只是"挑更好看的形式",不是拿到答案的必要条件。它不该有能力
    拖住一个已经算出来的正确答案 —— 单次 `sp.simplify` 就可能几分钟不返回。
    走 cas 闸口还有个附带好处:同一个变形超时过就不会被反复重试。
    """
    from . import cas

    plan = [
        ("展开", lambda e: sp.expand(e)),
        ("部分分式分解", lambda e: sp.apart(e, x)),
        ("通分约简", lambda e: sp.cancel(e)),
        ("因式分解", lambda e: sp.factor(e)),
        ("三角化简", lambda e: sp.trigsimp(e)),
        ("倍角展开", lambda e: sp.expand_trig(e)),
        ("改写成 sin/cos", lambda e: e.rewrite(sp.sin)),
        ("整体化简", lambda e: sp.simplify(e)),
    ]
    for name, fn in plan:
        status, new = cas.guard(f"rewrite|{name}|{sp.srepr(f)}",
                                lambda fn=fn: fn(f), REWRITE_TIMEOUT)
        if status != "ok" or new is None:
            continue
        if new != f and not new.has(sp.Integral):
            yield name, new


# 方法链(trace)是**解释性**元数据,不是答案的一部分。
# SymPy 的逐步积分器要在整个规则空间里搜索,实测 `∫sin 3x·cos 5x dx`
# 上要 **105 秒** —— 而它的答案 0.17 秒就出来了。
# 解释性工作绝不能拖住答案,所以单独限时;拿不到就给一条空链。
TRACE_TIMEOUT = 4.0


def step_trace_guarded(f: sp.Expr, x: sp.Symbol, max_depth: int = 12):
    """带超时的 `step_trace`。

    返回 (链, 是否识别出方法)。超时或出错时返回空链 ——
    它只影响报告里"方法链"那一栏,不影响任何求解结论。
    """
    status, payload = run_with_timeout(
        lambda: step_trace(f, x, max_depth), TRACE_TIMEOUT)
    if status == "ok" and payload is not None:
        return payload
    return [], False


def solve(f: sp.Expr, x: sp.Symbol, substitutions: list[Substitution] | None = None,
          max_attempts: int = 24, seed: int = 0,
          sink: ResultSink | None = None) -> Solution:
    """求解 ∫f dx,只返回通过验证的结果。

    策略按"先便宜后昂贵"的顺序尝试,并且做**结果择优**:
    一旦拿到干净的结果(ugliness 低于阈值)就立刻收工;
    如果结果虽然验证通过但很难看,就继续往下试,最后取最干净的那个。

    同一个代换换回 x 时往往有多个分支、多种写法(比如原样代入 vs 化简后),
    这些会**全部**验证一遍再挑最干净的那个 —— 因为 sp.simplify 有时候
    会把结果化简成 I*(log(...) - tanh(log(...))) 这种东西。

    sink:如果给了,每验证通过一个原函数就立刻放进去(而不是等这个函数 return)。
    这样即使调用方在外面套了超时、到点放弃了,也还能把已经找到的原函数取走 ——
    这是 timeout.py 里说的"病二"的解药。定积分那边正是靠它,
    才能在「牛顿-莱布尼茨」整条超时之后仍然拿到 F,再去算 F(b)−F(a)。
    """
    sol = Solution(ok=False, x=x, f=f)
    substitutions = substitutions or []
    best: Attempt | None = None
    best_ugly = float("inf")
    verified_at: float | None = None

    def record(attempt: Attempt) -> None:
        """登记一次尝试;如果验证通过并且更干净,就更新当前最优。"""
        nonlocal best, best_ugly, verified_at
        sol.attempts.append(attempt)
        if not (attempt.result is not None
                and attempt.verify is not None and attempt.verify.ok):
            return
        # ★ 成果在"算出来的当下"就交出去,不等最后 return
        if sink is not None:
            sink.put(attempt)
        if verified_at is None:
            verified_at = time.monotonic()
        ugly = ugliness(attempt.result)
        if ugly < best_ugly:
            best, best_ugly = attempt, ugly

    def clean() -> bool:
        """已经拿到足够干净的结果了吗?"""
        return best is not None and best_ugly <= CLEAN_THRESHOLD

    def budget_left() -> bool:
        if len(sol.attempts) >= max_attempts:
            if not any("达到尝试上限" in e for e in sol.errors):
                sol.errors.append(f"达到尝试上限 {max_attempts},提前停止")
            return False
        # 父层(比如定积分那边的「牛顿-莱布尼茨」12 秒)预算用完了就停手,
        # 不要再启动一次注定会被掐掉、而且成果也拿不回来的尝试。
        if budget_exhausted():
            if not any("父层时间预算" in e for e in sol.errors):
                sol.errors.append("父层时间预算已耗尽,提前停止")
            return False
        # 已经有通过验证的结果了 —— 后面的尝试只是在挑更好看的形式。
        # 那是增值工作,不能让它把"已经拿到的答案"拖住。
        if (verified_at is not None
                and time.monotonic() - verified_at > FORM_BUDGET):
            if not any("形式择优" in e for e in sol.errors):
                sol.errors.append(
                    f"已拿到通过验证的结果,形式择优超过 {FORM_BUDGET:.0f} 秒,"
                    "停止继续变形尝试(不影响结果正确性)")
            return False
        return True

    # --- 策略 1:直接积分
    record(_verified_attempt("直接积分(SymPy 内置算法)", f, x, f, x))
    if clean():
        return _finalize(sol, best, x)

    # --- 策略 2:恒等变形后积分(结果仍按原 f 验证)
    # 用 while + next() 而不是 for:生成器是惰性的,一次 next() 才去算下一个
    # 变形。必须**先查预算、再取下一个**,否则那个昂贵的变形已经算完了,
    # 预算检查就成了马后炮。
    rewrites = _rewrite_candidates(f, x)
    while budget_left():
        try:
            name, rewritten = next(rewrites)
        except StopIteration:
            break
        record(_verified_attempt(f"恒等变形:{name}", rewritten, x, f, x,
                                 note=f"变形为 {rewritten}"))
        if clean():
            return _finalize(sol, best, x)

    # --- 策略 3:按检索到的方法卡片做代换
    for sub in substitutions:
        if not budget_left():
            break
        label = f"代换:{sub.name}"
        try:
            transformed = _substitute(f, x, sub)
        except Exception as exc:
            sol.attempts.append(Attempt(label, f, x, None, None,
                                        f"代换过程出错:{type(exc).__name__}"))
            continue

        attempt_label = f"{label}(化为 ∫{transformed} d{sub.t})"
        intermediate, msg = _try_integrate(transformed, sub.t)
        if intermediate is None:
            sol.attempts.append(Attempt(attempt_label, transformed, sub.t, None, None, msg))
            continue

        candidates = _back_substitute(intermediate, sub, x)
        if not candidates:
            sol.attempts.append(Attempt(
                attempt_label, transformed, sub.t, intermediate, None,
                "积出来了,但反函数解不出来,无法换回 x",
                intermediate=intermediate, substitution=sub))
            continue

        # 这个代换的每个分支/每种写法都验证一遍,再挑最干净的
        for candidate in candidates:
            check = verify_antiderivative(candidate, f, x)
            record(Attempt(attempt_label, transformed, sub.t, candidate, check,
                           f"由 {sub.name} 得到;{sub.why}",
                           intermediate=intermediate, substitution=sub))
        if clean():
            return _finalize(sol, best, x)

    # --- 策略 4:manualintegrate 兜底
    if budget_left():
        try:
            from sympy.integrals.manualintegrate import manualintegrate
            manual = manualintegrate(f, x)
            if manual is not None and not manual.has(sp.Integral):
                check = verify_antiderivative(manual, f, x)
                record(Attempt("manualintegrate 兜底", f, x, manual, check,
                               "SymPy 的逐步积分器给出的结果"))
        except Exception as exc:
            sol.errors.append(f"manualintegrate 失败:{type(exc).__name__}")

    if best is not None:
        return _finalize(sol, best, x)
    # 拒答也要给出步骤:让"没做出来"是一个**有结论**的结果,
    # 而不是报告里一片空白(定积分那边的发散结论也是这么处理的)。
    sol.steps = _refusal_steps(f, x, sol.errors)
    return sol


def _verify_line(chosen: Attempt, x: sp.Symbol) -> str:
    """把验证结果写成人能读的一句话。

    `proof` 只是"用了哪条证据链"(常常就一个词 `symbolic`),
    真正有信息量的是 `detail`(数值交叉验证的点数/最大误差)。
    """
    check = chosen.verify
    if check is None:
        return f"验证:对结果求导回验 F′({x}) ≡ 被积函数 —— 未做验证"
    bits = [b for b in (check.proof, check.detail) if b]
    text = ";".join(bits)
    return f"验证:对结果求导回验 F′({x}) ≡ 被积函数 —— {text}"


def _describe_steps(chosen: Attempt, f: sp.Expr, x: sp.Symbol) -> list[str]:
    """把不定积分的解法渲染成人能读的步骤。

    ## 和定积分那条链的**关键区别**

    定积分的换元是**等值变换**(上下限同时换),所以内层的值就是原积分的值,
    **不存在回代**(见 README §6g.2)。不定积分正相反:换元之后求出来的
    原函数是在 `t` 里的,必须把 `t` 换成 `x` 才是答案 ——
    **回代这一步是不定积分链上不可省的一环**,也正是过去报告里缺掉的那一环
    (那时只给了一段"代换之后那个积分"的规则库路径,看得出方法,
    看不出它怎么接回原积分)。

    两条链形状不同,步骤就不能照抄 —— 这是同一个教训的第二次出现。
    """
    steps: list[str] = [f"原积分 ∫{f} d{x}"]

    sub = chosen.substitution
    if sub is not None:
        pieces = [f"换元:{getattr(sub, 'name', '换元')}"]
        x_expr = getattr(sub, "x_expr", None)
        t = getattr(sub, "t", None)
        if x_expr is not None and t is not None:
            pieces.append(f"令 x = {x_expr}(t = {t})")
        if getattr(sub, "card_id", None):
            pieces.append(f"依据卡片 {sub.card_id}")
        if getattr(sub, "why", ""):
            pieces.append(str(sub.why))
        steps.append(" · ".join(pieces) + f" ⟹ 化为 ∫{chosen.integrand} d{t}")

        if chosen.intermediate is not None:
            steps.append(f"求原函数:对代换后的积分求原函数,得到 "
                         f"{chosen.intermediate}(含变量 {t})")
        steps.append(f"回代:把 {t} 换回 {x} —— 得到 {chosen.result}"
                     "(不定积分**必须**做这一步,否则答案还留在代换变量里)")
    elif chosen.strategy.startswith("恒等变形"):
        # 恒等变形不是换元:积分变量没变,只是把被积函数改写成更好积的形式。
        # 这一步同样要说出来,否则读者看不出答案是从哪个式子积出来的。
        rewritten = chosen.note.replace("变形为", "").strip() or "—"
        steps.append(f"恒等变形:{chosen.strategy.split(':', 1)[-1]} —— 被积函数改写成 "
                     f"{rewritten}(变量仍是 {x},不需要回代)")
        steps.append(f"求原函数:对改写后的式子求原函数,得到 {chosen.result}")
    else:
        steps.append(f"求原函数:用「{chosen.strategy}」直接求,得到 {chosen.result}")

    steps.append(_verify_line(chosen, x))
    return steps


def _refusal_steps(f: sp.Expr, x: sp.Symbol, errors: list[str]) -> list[str]:
    """拒答时的步骤 —— 让它和"发散"一样是**有结论**的,而不是一片空白。"""
    steps = [
        f"原积分 ∫{f} d{x}",
        "结论:在尝试的路线里没有找到**能通过求导回验**的原函数,"
        "系统选择不作答 —— 而不是编一个看起来像样的答案给你。",
    ]
    for item in errors[:3]:
        steps.append(f"尝试记录:{item}")
    return steps


def _finalize(sol: Solution, chosen: Attempt, x: sp.Symbol) -> Solution:
    """把选中的那次尝试提升为最终答案,并抽取方法链。"""
    sol.ok = True
    sol.result = chosen.result
    sol.strategy = chosen.strategy
    sol.proof = chosen.verify.proof if chosen.verify else ""
    sol.detail = chosen.verify.detail if chosen.verify else ""

    trace, identified = step_trace_guarded(chosen.integrand, chosen.variable)
    sol.trace = trace
    sol.trace_var = chosen.variable
    sol.trace_identified = identified

    first_ok = next((a for a in sol.attempts
                     if a.result is not None and a.verify is not None and a.verify.ok), None)
    if first_ok is not None and first_ok is not chosen:
        note = (f"第一个通过验证的结果形式不佳(或更难读),"
                f"已改用「{chosen.strategy}」的等价结果")
        sol.detail = (sol.detail + ";" + note) if sol.detail else note
    sol.steps = _describe_steps(chosen, sol.f, x)
    return sol
