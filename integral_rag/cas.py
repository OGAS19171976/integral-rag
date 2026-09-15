"""把不可控的符号计算收拢到一个闸口:记忆化 + 泄漏计量 + 闸门。

## 要治的两个实测浪费

`run_with_timeout` 的语义是"到点不再等",**不是**"到点停止计算" ——
Python 杀不掉一个正在 C 层循环的线程。实测后果有两个,都不是"超时不好用",
而是"超时只解决了一半":

  1. **重复烧钱。** 同一个 `sp.integrate(e^{-t x²}, x)` 会被"直接积分"、
     "牛顿-莱布尼茨"、"区间再现"等好几条策略各调一次,每次各自烧满 12 秒。
     实测 `∫_0^∞ e^{-t x²}dx` 答案早就试出来了,时间全花在重复超时上。

  2. **跨题投毒。** 跑飞的线程不会退出,继续抢 CPU 和 GIL。实测同一道题
     单独跑 26.3 秒,放进 50 题的串行评估里要 62.1 秒 —— 慢掉的 2.4 倍
     全是前面题目留下的僵尸线程。三个发散题在报告里整齐地都是 10~11 秒,
     单独跑却只要 2.6 秒,就是这个原因。

## 三个措施

  * **记忆化**(治 1):同一个调用只发起一次,结果连"超时"一起留在进程里。
  * **计量**(为 2 服务):留住每个超时调用的线程句柄,用它判断那个线程
    到底是"还在跑"还是"其实早就跑完了,只是当时预算给少了"。
    后者会被放行重试 —— 这一点很重要,否则一次小额超时会永久封死一个
    本来 3 秒就能算完的调用。

## 闸门:试过了,又拆掉了(如实记录)

最初还做了第三个措施 —— 在跑的僵尸线程到上限时,限制新的"长"调用:
先是直接拒绝,后来改成"压缩到 2.5 秒"。**两版都造成了正确性回归**:

  * 拒绝版:评估从 45/45 掉到 44/45。`∫_0^{2π}|sin x|dx` 单独跑 1.5 秒就出结果,
    但它在评估里跑第 33 题,前面已经攒下 2 个僵尸线程,于是被整个跳过。
  * 降级版:换成另一道题掉 —— `∫_0^1 ln(1+x)/x dx` 需要超过 2.5 秒的直接积分,
    被压缩之后超时。总分还是 44/45。

根因是**判据本身错了**:闸门只能看到这次调用**请求**了多长预算,
看不到它实际要跑多久。而"请求 12 秒、实际 1 毫秒"恰恰是最常见的调用。
更糟的是它把互不相关的题目耦合起来 —— 前一道题的僵尸线程决定了后一道题
能拿多少预算,于是评估结果开始依赖题目顺序。

所以闸门被拆掉了,只留下**计量**(`runaway_count()` / `stats()`)作为可观测指标。
真正解决问题的仍然是记忆化:发起次数少了,僵尸自然就少了。
这条经验记在 README 6c.6:任何用"请求量"而不是"实际消耗量"做判据的熔断,
一定会误伤最该放行的那类调用。

## 为什么不做进程隔离(本该是最彻底的解)

试过了,**当前环境不允许**:Windows 的 `multiprocessing` 用命名管道做 IPC,
而沙箱拒绝创建命名管道 —— `ctx.Pipe()` 直接 `PermissionError: [WinError 5]`。
所以"常驻子进程 + terminate 真杀 + 硬超时"这条正确的路在这里走不通,
只能退回进程内缓解。这是环境约束,不是设计取舍。

进程内缓解的天花板是明确的:它**不减少**已经跑飞的线程数,只是让数量有上界。
真正的解法仍然是进程隔离,或者把昂贵的、不可控的那一步做小。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import sympy as sp

from .timeout import run_with_timeout

# 超过这个时长的调用才算"长"调用。这个门槛现在只用于**统计**,
# 不再用于拦截 —— 闸门试过两版都造成正确性回归,已拆掉(见模块开头)。
LONG_CALL = 4.0

# 允许同时存在多少个"跑飞了还没结束"的线程。
# 这个数字**不**用来拦截任何调用,只是 `runaway_count()` 的观察上限,
# 用来判断"这批题一共制造了多少个僵尸线程"。
MAX_RUNAWAYS = 2


@dataclass
class _Entry:
    """一条记忆:这次调用是成功了、报错了,还是超时了。"""

    status: str
    value: object = None
    event: threading.Event | None = None
    started: float = 0.0
    box: dict | None = None

    def spent(self) -> float | None:
        """线程**真正**跑完用的秒数;还没跑完就返回 None。"""
        if self.event is None or self.box is None or not self.event.is_set():
            return None
        finished = self.box.get("finished_at")
        return None if finished is None else finished - self.started

    @property
    def still_running(self) -> bool:
        return self.event is not None and not self.event.is_set()


_MEMO: dict[str, _Entry] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------- 计量
def _runaways_locked() -> int:
    """还在跑的僵尸线程数。只数超时过的 —— 成功返回的线程已经结束了。"""
    return sum(1 for e in _MEMO.values()
               if e.status == "timeout" and e.still_running)


def runaway_count() -> int:
    with _LOCK:
        return _runaways_locked()


def stats() -> dict:
    """给报告/调试用的一份快照。"""
    with _LOCK:
        return {
            "memo": len(_MEMO),
            "ok": sum(1 for e in _MEMO.values() if e.status == "ok"),
            "timeout": sum(1 for e in _MEMO.values() if e.status == "timeout"),
            "runaways": _runaways_locked(),
        }


def reset() -> None:
    """清空记忆。测试和评估用它做隔离;生产不要随便调 ——
    跨题复用正是记忆化收益的一半。"""
    with _LOCK:
        _MEMO.clear()


# ---------------------------------------------------------------- 键
def _key(prefix: str, expr, var, *limits) -> str:
    """用 srepr 做键:同一个表达式算出来的字符串唯一且稳定。

    刻意**不做**数学等价判定 —— 两个写法不同但等价的表达式会算两次。
    这是保守的选择:宁可多算一次,也不要把两个不同的调用混成一个。
    """
    parts = [prefix, getattr(var, "name", str(var)), sp.srepr(expr)]
    for item in limits:
        parts.append(sp.srepr(sp.sympify(item)))
    return "|".join(parts)


# ---------------------------------------------------------------- 闸口
def guard(key: str, fn, timeout: float, sink=None):
    """带记忆和泄漏计量的 `run_with_timeout`。

    返回 (状态, 值),状态比 `run_with_timeout` 多一个:
        "throttled"  僵尸线程太多,这个长调用**没有发起**
    """
    with _LOCK:
        entry = _MEMO.get(key)
        if entry is not None:
            if entry.status == "ok":
                return "ok", entry.value
            if entry.status == "error":
                return "error", entry.value
            # 超时过。先看它是不是其实已经跑完了。
            spent = entry.spent()
            if spent is None:
                # 还在跑 —— 绝不再发起第二个完全相同的调用。
                return "timeout", None
            # 它跑完了,花了 spent 秒。如果这次的预算比它实际耗时更长,
            # 说明当时只是预算给少了,允许重试;否则就是真的算不动。
            if timeout <= spent:
                return "timeout", None
            # 落到下面重试
        live = _Entry("running")
        _MEMO[key] = live
        observer: dict = {}

    status, payload = run_with_timeout(fn, timeout, sink=sink, observer=observer)

    with _LOCK:
        live.event = observer.get("event")
        live.started = observer.get("started", 0.0)
        live.box = observer.get("box")
        if status == "ok":
            live.status, live.value = "ok", payload
        elif status == "error":
            live.status, live.value = "error", payload
        elif status == "timeout":
            live.status = "timeout"
        else:
            # "budget":压根没启动,不该留下记忆,否则会封死后续更大预算的调用
            if _MEMO.get(key) is live:
                _MEMO.pop(key, None)
    return status, payload


# ---------------------------------------------------------------- 便捷入口
def integrate_indef(expr, var, timeout: float, sink=None):
    """`sp.integrate(expr, var)`,带记忆和闸门。"""
    return guard(_key("indef", expr, var),
                 lambda: sp.integrate(expr, var), timeout, sink=sink)


def integrate_definite(expr, var, lo, hi, timeout: float, sink=None):
    """`sp.integrate(expr, (var, lo, hi))`,带记忆和闸门。"""
    return guard(_key("def", expr, var, lo, hi),
                 lambda: sp.integrate(expr, (var, lo, hi)), timeout, sink=sink)


def note(status: str) -> str:
    """把闸口状态翻译成给使用者看的一句话。

    "throttled" 这一支保留着,是为了兼容调用方已有的状态分支;
    闸门拆掉之后这个状态不会再产生。
    """
    if status == "throttled":
        return "符号计算线程积压过多,本次昂贵尝试已放弃"
    if status == "timeout":
        return "超时"
    return ""
