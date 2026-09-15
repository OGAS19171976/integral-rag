"""给不可控的符号计算加超时,并且让超时**可以组合**。

SymPy 的 `integrate` 没有超时机制,个别积分会让它陷进去出不来 ——
实测 `∫_0^1 ln(1+x)/(1+x²) dx` 就是,`sp.integrate` 十几分钟不返回。
这类调用必须能被打断,否则整个系统会被一道题拖死。

## 为什么用守护线程,而不是子进程

子进程本来更彻底(能真正杀掉),但在这个场景下有两个硬伤:

  1. Windows 上 multiprocessing 用 spawn 启动,子进程会重新导入 `__main__`。
     调用方的脚本若没有 `if __name__ == "__main__"` 保护,就会无限递归地
     创建进程 —— 这是个很容易踩的坑,不该把这种负担甩给使用者。
  2. 每次调用都要把 SymPy 表达式 pickle 过去再 pickle 回来。

守护线程没有这些问题,而且几乎零启动开销,所以可以给每一处可疑的
符号计算都套上一层。

## 代价(以及后来怎么把它消掉的)

超时之后那个线程原本**还在后台跑** —— "到点不再等"并不等于"到点停止计算"。
它是 daemon,不阻止进程退出,但会一直占着一个核和 GIL。实测的后果不是"多花点电",
而是**拖慢后面每一道题**:同一道题单独跑 26.3 秒,放进 50 题的串行评估里要 62.1 秒。

后来发现 CPython 其实有一条**能真的打断它**的路:
`PyThreadState_SetAsyncExc(thread_id, exc)` 可以往目标线程异步注入一个异常,
异常会在下一个字节码边界抛出。sympy 的 `integrate` 几乎全是 Python 层循环,
所以这条注入**真的能把跑飞的计算停下来** —— 实测那个"十几分钟不返回"的
`∫ln(1+x)/(1+x²)dx` 被打断了。

现在 `run_with_timeout` 超时后会**尝试杀掉**那个线程,而不是丢下它跑:

    "timeout" 状态的语义因此变成"到点停手,并且尽力把线程也停下"。

**这不是硬保证**(见 `_kill_thread`):注入只能在字节码边界生效,
如果线程卡在真正不返回 C 代码里(比如装了 gmpy2 之后的大整数循环),
它照样杀不掉;那时行为退回成原来的"丢下不管",不会更糟。
要**硬保证**只有进程隔离一条路,而那是可行的 ——
本沙箱拒绝的只是**命名管道**这一种 IPC 传输,localhost TCP 实测完全可用
(见 README §6f)。

---

## 超时套了三层之后,会得两个病

"每层都套一个超时"看起来是防御性编程,实际上几乎必然踩到下面两个问题。
它们都不是"超时机制不好用",而是**用法错了**。

### 病一:预算不传播,内层的超时是假的

原来的代码里,`definite.py` 给一条策略 12 秒,而这条策略内部调用的
`solve.py` 给每一次 `sp.integrate` 20 秒。外层 12 秒先开火,
内层那 20 秒**永远用不到** —— 内层的超时形同虚设,内层"以为还有 20 秒、
于是继续尝试下一种方法"的判据也完全错判。

这是最隐蔽的一种:每个数字单看都合理,组合起来却没有意义。

**解法:预算向下传播。** 内层取「自己想要的」和「父层还剩多少」的较小值;
父层只剩 3 秒,内层就不可能拿到 20 秒。

### 病二:外层到点放弃时,内层已经算出来的成果被一起丢掉

`run_with_timeout` 原来的做法是"到点就不再等,把结果盒子扔掉"。
如果被放弃的那段计算内部**已经算出了一些有用的东西**,那些也一起没了。
实测 `∫_0^1 ln(1+x)/(1+x²)dx` 就死在这里:内层其实已经把
`ln(1+tan t)` 积出来了,外层一句「换元换限 超时」就把成果扔了。

**解法:成果在算出来的当下就交出来,别等最后 return。**
`ResultSink` 是一个线程安全的中转站:内层每验证通过一个结果就立刻放进去;
外层即使到点放弃,调用方仍然能把已经放进去的成果取走。

### 一般原则

这两条合起来是一句话:

    **超时的边界,必须和你"愿意丢掉的东西"的粒度对齐。**

想保住更细粒度的成果,就得让成果穿过边界流出来(病二的解法),
或者把边界往里挪到"丢掉也不可惜"的那一层 ——
比如把递归从带超时的策略内部挪出去,交给顶层队列调度。
只加超时、不做这两件事,内层的成果注定被丢。

### 边界:什么情况下救不回来

如果被放弃的是一整个**不可分割**的操作(一次永不返回的 `sp.integrate`),
那中间没有任何东西可以抢救。**你能保住的东西,粒度不可能比检查点更细。**
所以设计上要做的是:让昂贵的、不可控的那一步尽可能"小",
把它夹在两次廉价的检查点之间。
"""

from __future__ import annotations

import contextvars
import ctypes
import threading
import time
from dataclasses import dataclass

DEFAULT_TIMEOUT = 20.0

# 注入异常之后,等多久确认线程真的退出了。只是"确认"用的短等待,
# 不是新的预算 —— 退不出去也不比原来更糟(它本来就是 daemon)。
_KILL_GRACE = 0.5


class _Abandoned(BaseException):
    """用来打断跑飞线程的信号。

    **刻意继承 `BaseException` 而不是 `Exception`**:sympy 内部到处都是
    `except Exception`,如果继承 Exception,信号会被就地吞掉、线程继续跑,
    这次注入就白做了。走 BaseException 才能穿过那些宽泛的 except。
    """


def _kill_thread(thread: threading.Thread) -> bool:
    """往目标线程异步注入 `_Abandoned`,让它真的停下来。

    CPython 的 `PyThreadState_SetAsyncExc` 会在目标线程的**下一个字节码边界**
    抛出指定异常。sympy 的 `integrate` 几乎全是 Python 层循环,所以这条注入
    通常有效 —— 实测能打断那个十几分钟不返回的积分。

    ## 为什么这不是硬保证(如实说明)

    * 异常只能在**字节码边界**抛出。线程若卡在真正不返回的 C 代码里
      (例如装了 gmpy2 之后的大整数循环),注入会被推迟到它返回 Python 为止,
      等于杀不掉。此时调用方退回"丢下不管"的老行为,不会更糟。
    * 打断是**任意位置**的。理论上可能在别人持有锁时把它带走 ——
      但 CPython 的 `with` / `try-finally` 在异常传播时照常释放锁,
      sympy 的全局缓存用的是 `with _cache_lock`,而且缓存是"算完才写入",
      被中断只会少一条缓存,不会写进错的值。
      `tests/test_all.py` 的 H 组专门验证了"杀完之后进程仍然可用"。
    * 因此定位是**尽力而为的清理**,不是资源隔离。要硬保证请用进程隔离。
    """
    ident = thread.ident
    if ident is None:
        return False
    try:
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(ident), ctypes.py_object(_Abandoned))
    except Exception:                                 # noqa: BLE001
        return False
    if result == 1:
        return True
    if result > 1:
        # CPython 文档的要求:返回 >1 说明影响到了多个线程,必须立即撤销
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(ident), None)
        except Exception:                             # noqa: BLE001
            pass
    return False


# 供诊断/测试观察:一共超时了多少次、其中多少次把线程家族都停下了。
_KILL_LEDGER: list[bool] = []


# ================================================================ 线程家族
# ★ 这是"杀不掉跑飞线程"的**结构性**修法。
#
# 超时是**分层**的:外层策略一个 12 秒闸,内层 integrate 一个 20 秒闸。
# 外层到点时,外层线程正阻塞在内层 `done.wait()` 上 —— 对它注入异常毫无作用
# (CPython 只能在**字节码边界**抛,阻塞在锁等待里的线程到不了边界);
# 而**真正在烧 CPU 的内层线程,外层根本不知道它的存在**,于是从来没人去杀它。
# 结果就是:外层"超时"了,里面那个计算继续跑满一个核。
#
# 解法:同一个调用链上产生的线程登记进同一个**家族**,超时时对家族里的
# 每一个成员注入异常。**杀掉的范围必须和放弃的范围一致。**
_family: contextvars.ContextVar[set | None] = contextvars.ContextVar(
    "dsh_thread_family", default=None)


def _stop_family(family: set, grace: float) -> bool:
    """把家族里还活着的线程全部注入异常并等一会儿。

    返回是否**全部**停下。没停下的那些通常是"阻塞在内层等待上"的外层线程 ——
    它们不烧 CPU,等内层结束后会自己退出,所以不是性能问题(但如实报出来)。
    """
    victims = [t for t in list(family) if t.is_alive()]
    # 顺手把已经结束的成员剔掉,免得家族在一次长调用里无限膨胀
    family.intersection_update(victims)
    for victim in victims:
        _kill_thread(victim)
    deadline = time.monotonic() + grace
    for victim in victims:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            victim.join(remaining)
    return not any(t.is_alive() for t in victims)


def kill_stats() -> dict:
    """超时清理的统计。

    `stopped` 是"超时之后确认线程家族都停了"的次数。
    `still_running` 是没清理干净的 —— 那说明有线程卡在注入够不着的地方
    (典型是阻塞在内层等待上),行为退回到"丢下不管"。
    这个比例本身就是这套机制可靠性的度量。
    """
    return {
        "timeouts": len(_KILL_LEDGER),
        "stopped": sum(1 for ok in _KILL_LEDGER if ok),
        "still_running": sum(1 for ok in _KILL_LEDGER if not ok),
    }


# ================================================================ 截止时间
@dataclass(frozen=True)
class Deadline:
    """一个可以在调用链上向下传播的截止时刻(用单调时钟)。"""

    expires_at: float

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.expires_at <= time.monotonic()

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(time.monotonic() + max(0.0, seconds))


# 用 ContextVar 而不是全局变量:每个线程各自持有自己的截止时间,
# 而子线程通过 contextvars.copy_context() 继承父线程的那一份。
_deadline: contextvars.ContextVar[Deadline | None] = contextvars.ContextVar(
    "dsh_deadline", default=None)


def current_deadline() -> Deadline | None:
    return _deadline.get()


def remaining_budget(requested: float) -> float:
    """把「我想要的预算」和「父层还剩多少」取较小者。

    这就是病一的解药。没有父层时就是 requested 本身。
    """
    parent = _deadline.get()
    if parent is None:
        return requested
    return min(requested, parent.remaining())


def budget_exhausted() -> bool:
    """父层给的预算是不是已经用完了。长循环里应该拿它当退出条件。"""
    parent = _deadline.get()
    return parent is not None and parent.expired


# ================================================================ 成果中转站
class ResultSink:
    """线程安全的成果中转站 —— 病二的解药。

    内层每算出一个**有用**的东西就立刻 put 进来,而不是攒到最后 return。
    外层超时放弃之后,调用方照样能 drain 出已经放进去的成果。

    只保留前 `limit` 个,免得一次搜索往里灌几百条中间结果。
    """

    def __init__(self, limit: int = 8):
        self._items: list = []
        self._lock = threading.Lock()
        self._limit = limit

    def put(self, item) -> bool:
        with self._lock:
            if len(self._items) >= self._limit:
                return False
            self._items.append(item)
            return True

    def drain(self) -> list:
        with self._lock:
            items, self._items = self._items, []
            return items

    def peek(self) -> list:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# ================================================================ 入口
def run_with_timeout(fn, timeout: float = DEFAULT_TIMEOUT,
                     sink: ResultSink | None = None,
                     observer: dict | None = None):
    """在守护线程里跑 `fn()`。返回 (状态, 值)。

    状态取值:
        "ok"      正常返回,值在第二个位置
        "timeout" 超时,值恒为 None
        "budget"  父层预算已经耗尽,压根没启动
        "error"   抛了异常,值就是那个异常对象

    sink 可选,但**强烈建议给**:内层可以在算出来的当下把成果 put 进去。
    无论返回什么状态,调用方都应该顺手 drain 一次 ——
    里面可能已经有内层在超时前抢出来的成果。

    observer 可选,用来**在超时之后继续追踪那个线程**。超时只是不再等,
    线程还在跑;想知道它到底跑完了没有,只能留着它的 `done` 事件。
    传进来的字典会被填上:
        "event"    那个 `threading.Event`;set 了就是线程已结束
        "started"  线程启动时刻(单调时钟)
        "box"      结果盒子,里面可能有 "finished_at" / "value" / "error"
    有了它,`cas.py` 才能区分"还在跑"和"其实早跑完了,只是当时预算不够"。
    """
    # 病一:预算向下传播。父层只剩 3 秒,这里就不可能拿到 20 秒。
    budget = remaining_budget(timeout)
    if budget <= 0:
        return "budget", None

    box: dict = {}
    done = threading.Event()

    # 本次调用属于哪个线程家族。父层没有家族时(比如直接从主线程调用)
    # 就地新建一个 —— 这样并列的两次调用互不牵连,一次超时不会去杀
    # 上一次已经返回的那批线程。
    family = _family.get()
    if family is None:
        family = set()

    def target() -> None:
        # 子线程自己的截止时间 = 它实际拿到的预算
        _deadline.set(Deadline.after(budget))
        # ★ 把家族传下去:这一支里再起的线程(内层的 run_with_timeout)
        # 会登记到同一个家族,于是超时时**一起**被杀 —— 见 _family 的注释。
        _family.set(family)
        try:
            box["value"] = fn()
        except BaseException as exc:          # noqa: BLE001 —— 什么都得接住
            box["error"] = exc
        finally:
            box["finished_at"] = time.monotonic()
            done.set()

    # copy_context() 让子线程继承父线程的截止时间,这样更内层的调用
    # 也**看得见**这一层的剩余预算,而不是各自为政。
    context = contextvars.copy_context()
    thread = threading.Thread(target=context.run, args=(target,), daemon=True)
    family.add(thread)
    started = time.monotonic()
    thread.start()

    if observer is not None:
        observer["event"] = done
        observer["started"] = started
        observer["box"] = box

    if not done.wait(budget):
        # ★ 到点不再等,而且**把整支线程家族都停下** —— 见 _kill_thread / _family。
        # 只杀直接子线程是不够的:外层线程往往正阻塞在内层 `done.wait()` 上,
        # 注入对它无效,而真正烧 CPU 的是内层那个线程。
        #
        # 判据用"还活着吗",不要用注入函数的返回值:线程在超时边界上恰好
        # 自己跑完时,注入会因为线程状态已消失而失败 —— 那是**成功**。
        stopped = _stop_family(family, _KILL_GRACE)
        _KILL_LEDGER.append(stopped)
        if observer is not None:
            observer["stopped"] = stopped
        return "timeout", None
    if "error" in box:
        return "error", box["error"]
    return "ok", box.get("value")
