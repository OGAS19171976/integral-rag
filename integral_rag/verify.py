"""验证层 —— 整个系统可信度的来源。

一个积分结果要么被证明是对的,要么就不能用。这里用两条独立的证据链:

  1) 符号验证:化简 F'(x) − f(x) 是否恒等于 0。
     快,但 SymPy 的 simplify 并不完备 —— 化简不出来不等于错。
  2) 数值验证:在若干个随机点上比较 F'(x) 和 f(x) 的高精度数值。
     慢一点,但对"化简不出来的恒等式"很有用,而且能抓住更隐蔽的错误:
       · 分支选取错误(该用 arccos 的地方用了 arcsin 的另一个分支)
       · 常数项写错、符号写反
       · 参数条件漏掉(比如结果只在 x>a 时成立)

关于"定义域不连通"的处理:
    √(x²−a²) 只在 |x|>a 上有实值,于是 x>a 和 x<−a 是两个互不相干的区间。
    数学上原函数只要求在**一个区间**上满足 F'=f,所以这里允许
    "某一个连通分支上的采样点全部通过"。这不会放过错误答案 ——
    如果 F'−f 不恒等于 0,它在任何区间上都不会恒为 0。
    这种情况会在报告里如实标注"只在某一侧验证通过"。

凡是进入最终答案的原函数,必须至少通过其中一条。这条规则不能松 ——
大模型提出的方法、CAS 吐出的结果、检索到的方法卡片,全都得过这一关。

实测价值:∫√(x²−a²)/x dx 直接用 SymPy 积出来的 Piecewise 主分支,
导数符号是反的(在 x=−3.7 处 F' = +0.835 而 f = −0.835)。
这条错误正是被数值验证抓出来的。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import sympy as sp

# 采样点池:横跨负半轴、原点附近、正半轴,并且特意在 |x| 较大和较小的
# 区域都放了点 —— 这样 sqrt(a²−x²) 和 sqrt(x²−a²) 这两类能分别在
# "内侧"和"外侧"找到足够多的实数定义域采样点。
_X_POOL = (
    -7.4, -5.5, -3.7, -2.4, -1.6, -1.15, -0.9, -0.62, -0.35,
    0.32, 0.73, 1.1, 1.35, 1.8, 2.6, 3.9, 5.5, 7.4,
)
_POS_POOL = (2.0, 3.0, 1.4, 5.0, 2.5)
_NEG_POOL = (-1.7, -2.3, -0.8, 1.9, -3.0)

_TOL = 1e-7          # 相对误差容限
_MIN_POINTS = 3      # 至少要有几个有效采样点才敢下结论


@dataclass
class VerifyResult:
    ok: bool
    proof: str                  # 说明用了哪条证据链
    detail: str = ""            # 失败时的具体原因 / 成功时的补充说明
    n_points: int = 0
    worst_error: float = 0.0


# ---------------------------------------------------------------- 工具
def _eval_numeric(expr: sp.Expr, subs: dict, dps: int = 25):
    """在给定代换下求高精度数值,返回 complex 或 None(无法取值)。"""
    try:
        value = expr.subs(subs)
        if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.AccumBounds):
            return None
        c = complex(sp.N(value, dps))
    except Exception:
        return None
    if not (math.isfinite(c.real) and math.isfinite(c.imag)):
        return None
    return c


def _assignments(params: list[sp.Symbol], seed: int) -> list[dict]:
    """给参数造几组不同的取值,避免只在一组参数上碰巧成立。"""
    rng = random.Random(seed)
    out = []
    for i in range(4):
        assign = {}
        for j, sym in enumerate(params):
            if sym.is_positive:
                v = _POS_POOL[(i + j) % len(_POS_POOL)]
                if sym.is_integer:
                    v = float(int(v) + 1)
            elif sym.is_real:
                pool = _POS_POOL if (i + j) % 2 == 0 else _NEG_POOL
                v = pool[(i * 2 + j) % len(pool)]
            else:
                v = _POS_POOL[(i + j) % len(_POS_POOL)]
            assign[sym] = sp.Float(v + rng.uniform(-0.05, 0.05), 25)
        out.append(assign)
    return out


def _unwrap_piecewise(expr: sp.Expr) -> sp.Expr:
    """Piecewise 取默认分支(最后那个条件为 True 的分支)来做验证。

    这是有损的:如果默认分支之外还有别的分支,验证不到。所以调用方要
    在报告里标注"结果含分段函数,只验证了主分支"。"""
    if isinstance(expr, sp.Piecewise) and expr.args:
        last = expr.args[-1]
        if len(last) == 2 and last[1] is sp.true:
            return last[0]
    return expr


# ---------------------------------------------------------------- 符号链
# 有意做成"带名字的阶梯":报告里要能写出**凭什么说它是对的**。
# 只回一个 True,报告就只能印一句 `symbolic` —— 那等于没说。
_SYMBOLIC_LADDER = (
    ("simplify", lambda e: sp.simplify(e)),
    ("trigsimp∘expand_trig", lambda e: sp.trigsimp(sp.expand_trig(e))),
    ("cancel", lambda e: sp.cancel(e)),
    ("radsimp", lambda e: sp.radsimp(e)),
    ("expand_trig∘expand 后 simplify",
     lambda e: sp.simplify(sp.expand_trig(sp.expand(e)))),
    ("together", lambda e: sp.together(e)),
)


def _symbolic_verify(F: sp.Expr, f: sp.Symbol, x: sp.Symbol) -> str | None:
    """符号上验证 F′ = f。返回**成功的那一级的名字**;全失败返回 None。"""
    g = sp.expand(sp.diff(F, x) - f)
    if g == 0:
        return "求导后的差式直接为 0"
    for name, step in _SYMBOLIC_LADDER:
        try:
            if step(g) == 0:
                return f"求导后的差式经 {name} 化简为 0"
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- 数值链
def _numeric_verify(F: sp.Expr, f: sp.Expr, x: sp.Symbol, seed: int = 0) -> VerifyResult:
    """在采样点上核对 F'(x) 与 f(x)。

    点按 x 的符号分桶(对应"定义域的不同连通分支"),允许某一个分支
    整体通过即可 —— 理由见模块开头的说明。
    """
    dF = sp.diff(F, x)
    params = sorted((f.free_symbols | dF.free_symbols) - {x}, key=lambda s: s.name)

    # 桶内容:(是否通过, 相对误差, 说明)
    buckets: dict[int, list[tuple[bool, float, str]]] = {}

    for assign in _assignments(params, seed):
        for xv in _X_POOL:
            subs = dict(assign)
            subs[x] = sp.Float(xv, 25)

            fv = _eval_numeric(f, subs)
            if fv is None:
                continue
            # 被积函数在这一点的取值本身不是实数 → 不在实数定义域,跳过
            if abs(fv.imag) > 1e-9 * max(1.0, abs(fv)):
                continue

            dv = _eval_numeric(dF, subs)
            if dv is None:
                continue

            side = 1 if xv > 0 else -1
            bucket = buckets.setdefault(side, [])

            # f 是实数,但导数出现了不可忽略的虚部 → 原函数分支选错了
            if abs(dv.imag) > 1e-6 * max(1.0, abs(dv)):
                bucket.append((False, float("inf"),
                               f"x={xv:.3f} 处原函数导数的虚部为 {dv.imag:.3e},分支选取错误"))
                continue

            scale = max(1.0, abs(fv))
            err = abs(dv - fv) / scale
            bucket.append((err <= _TOL, err,
                           f"x={xv:.3f} 处 F'(x)={dv.real:.10g} 但 f(x)={fv.real:.10g},"
                           f"相对误差 {err:.3e}"))

    all_points = [p for points in buckets.values() for p in points]

    verdict = _judge(all_points, "全部采样点")
    if verdict is not None:
        return verdict

    # 逐连通分支试:原函数只需要在某一个区间上成立
    for side, points in sorted(buckets.items()):
        label = "x>0 一侧" if side > 0 else "x<0 一侧"
        verdict = _judge(points, label)
        if verdict is not None:
            other = [p for s, ps in buckets.items() if s != side for p in ps]
            if any(not ok for ok, _, _ in other):
                verdict.detail = (f"只在 {label} 的 {verdict.n_points} 个采样点上通过验证;"
                                  f"原函数定义域不连通,另一侧需要不同的积分常数")
            return verdict

    # 两种判定都不成立,给出信息量最大的那条失败原因
    failures = [msg for ok, _, msg in all_points if not ok]
    if failures:
        return VerifyResult(ok=False, proof="numeric", detail=failures[0],
                            n_points=len(all_points),
                            worst_error=max((e for _, e, _ in all_points), default=0.0))
    return VerifyResult(ok=False, proof="numeric",
                        detail=f"只找到 {len(all_points)} 个有效采样点,样本不足,无法确认",
                        n_points=len(all_points))


def _judge(points: list[tuple[bool, float, str]], label: str) -> VerifyResult | None:
    """某一组采样点能否支持"这是一个原函数"。通过返回结果,否则返回 None。"""
    if len(points) < _MIN_POINTS:
        return None
    bad = [msg for ok, _, msg in points if not ok]
    if bad:
        return None
    worst = max(err for _, err, _ in points)
    return VerifyResult(
        ok=True,
        proof=f"numeric({len(points)} points on {label}, worst rel err {worst:.2e})",
        n_points=len(points),
        worst_error=worst,
    )


# ---------------------------------------------------------------- 主入口
def verify_antiderivative(F: sp.Expr | None, f: sp.Expr, x: sp.Symbol,
                          seed: int = 0) -> VerifyResult:
    """验证 F 是不是 f 关于 x 的原函数。"""
    if F is None:
        return VerifyResult(ok=False, proof="none", detail="没有候选原函数")

    if not isinstance(F, sp.Expr):
        return VerifyResult(ok=False, proof="none", detail=f"结果类型异常:{type(F).__name__}")

    if F.has(sp.Integral):
        # 没积出来,SymPy 原样返回了 Integral(...)
        return VerifyResult(ok=False, proof="none", detail="结果是未求值的 Integral,即未能积出")

    if F.has(sp.DiracDelta, sp.Heaviside) and not f.has(sp.DiracDelta, sp.Heaviside):
        return VerifyResult(ok=False, proof="none", detail="结果引入了被积函数中不存在的广义函数")

    if F.is_number and f.has(x):
        return VerifyResult(ok=False, proof="none", detail="结果与 x 无关,显然不是本问题的原函数")

    piecewise_note = "结果含分段函数,只在主分支上通过验证" if F.has(sp.Piecewise) else ""
    probe = _unwrap_piecewise(F)

    route = _symbolic_verify(probe, f, x)
    if route:
        return VerifyResult(ok=True, proof=f"符号验证:{route}", detail=piecewise_note)

    result = _numeric_verify(probe, f, x, seed=seed)
    if piecewise_note and result.ok:
        result.detail = f"{piecewise_note};{result.detail}" if result.detail else piecewise_note
    return result
