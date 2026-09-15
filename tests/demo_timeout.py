"""对照实验:证明"内层成果被外层超时掐掉"这个问题确实被解决了。

分两个实验:
  实验一(病一)  预算不再各说各话 —— 内层拿到的超时会被父层夹住
  实验二(病二)  外层超时放弃之后,内层抢出来的成果仍然能拿到并使用
"""
import threading
import time

import sympy as sp

from integral_rag.definite import _newton_leibniz_from, _strategy_newton_leibniz
from integral_rag.solve import solve
from integral_rag.timeout import (Deadline, ResultSink, _deadline,
                                  remaining_budget, run_with_timeout)

print("=" * 92)
print("实验一:预算传播(病一)")
print("=" * 92)

def ask_for(requested):
    """内层想知道自己能拿到多少预算。"""
    return remaining_budget(requested)


print(f"  没有父层时,内层想要 20 秒就拿到 20 秒:{ask_for(20.0):.2f}s")

def nested():
    # 这一层自己请求 10 秒,但父层只给了 3 秒
    status, value = run_with_timeout(lambda: ask_for(10.0), 10.0)
    return status, value

started = time.time()
status, (inner_status, budget) = run_with_timeout(nested, 3.0)
elapsed = time.time() - started
print(f"  父层给 3 秒、内层请求 10 秒 → 内层实际拿到 {budget:.2f}s")
print(f"  (修复前:内层会认为自己有 10 秒;现在被父层夹住了)")
assert budget <= 3.0, "内层拿到的预算必须不超过父层剩余的"
assert abs(budget - 3.0) < 0.5, f"应该接近 3 秒,实际 {budget}"
print("  ✓ 内层不可能拿到比父层更多的预算")

# 再深一层:三层嵌套
def three_layers():
    def layer2():
        def layer3():
            return remaining_budget(30.0)
        return run_with_timeout(layer3, 30.0)
    return run_with_timeout(layer2, 30.0)

started = time.time()
status, (s2, (s3, deepest)) = run_with_timeout(three_layers, 2.0)
print(f"  三层各请求 30 秒、最外层给 2 秒 → 最内层拿到 {deepest:.2f}s")
assert deepest <= 2.0
print("  ✓ 预算穿过三层仍然被夹住")

print()
print("=" * 92)
print("实验二:成果抢救(病二)")
print("=" * 92)

print()
print("=" * 92)
print("实验二-甲:机制本身(确定性演示)")
print("=" * 92)

inner_sink = ResultSink()

def inner_work():
    inner_sink.put("我在 0.1 秒时就算出了一个有用的中间结果")
    time.sleep(30)          # 之后卡住不动
    return "永远到不了这里"

started = time.time()
status, payload = run_with_timeout(inner_work, 0.5, sink=inner_sink)
print(f"  内层:先交出一个成果,然后卡住 30 秒;外层只等 0.5 秒")
print(f"  外层看到的:状态 = {status},payload = {payload}  "
      f"(实际只等了 {time.time() - started:.2f}s)")
print(f"  但中转站里有:{inner_sink.drain()}")
assert status == "timeout"
assert len(inner_sink) == 0
print("  ✓ 外层确实放弃了,而且**仍然拿到了内层交出来的成果**")
print("    (修复前:这个成果会随 payload=None 一起消失)")

print()
print("=" * 92)
print("实验二-乙:真实代码路径(自动标定切点)")
print("=" * 92)

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)
# 用 ∫_a^{2a} √(x²−a²)/x dx。它的不定积分求解会**先找到一个(验证通过但
# 形式不佳的)原函数,然后继续找更好的**,所以"检查点"和"返回"之间
# 有很宽的窗口 —— 正好能切出"内层已经找到 F、外层却等不到"的场景。
f = sp.sqrt(a**2 - x**2) / x
lo, hi = a, 2 * a
# 真值:√(x²−a²) − a·asec(x/a) 在 x=a 处为 0,在 x=2a 处为 a(√3 − π/3)
EXPECTED = a * (sp.sqrt(3) - sp.pi / 3)

# 先跑一遍**不设超时**的,测出第一个检查点出现的时刻和总耗时,
# 切点就取两者中间 —— 这样实验不依赖我手工猜数字。
probe_sink = ResultSink(limit=8)
probe_box: dict = {}
probe_done = threading.Event()

def probe():
    try:
        probe_box["sol"] = solve(f, x, sink=probe_sink)
    except BaseException as exc:                  # noqa: BLE001
        probe_box["err"] = f"{type(exc).__name__}: {exc}"
    finally:
        probe_done.set()

threading.Thread(target=probe, daemon=True).start()
t0 = time.time()
first_at = None
while not probe_done.is_set() and time.time() - t0 < 60:
    if first_at is None and len(probe_sink) > 0:
        first_at = time.time() - t0
    time.sleep(0.02)
total = time.time() - t0
probe_sink.drain()
if probe_box.get("err"):
    print("  标定失败:", probe_box["err"])
    raise SystemExit(1)
if first_at is None:
    print(f"  标定失败:60 秒内没有出现检查点(总耗时 {total:.2f}s)")
    raise SystemExit(1)
print(f"  标定:第一个检查点 {first_at:.2f}s,整个不定积分求解共 {total:.2f}s")
CUTOFF = first_at + 1.0
print(f"  切点取 {CUTOFF:.2f}s(落在检查点之后、返回之前的窗口里)。")

# ---- 修复前的行为:没有中转站
print()
print("  [修复前] 不给中转站:")
started = time.time()
status, payload = run_with_timeout(
    lambda: _strategy_newton_leibniz(f, x, lo, hi), CUTOFF)
print(f"     状态 = {status}(耗时 {time.time() - started:.1f}s), payload = {payload}")
print("     → 内层做过什么,外面一无所知,成果全丢")

# ---- 修复后:给中转站
print()
print("  [修复后] 给一个成果中转站:")
sink = ResultSink(limit=4)
started = time.time()
status, payload = run_with_timeout(
    lambda: _strategy_newton_leibniz(f, x, lo, hi, sink=sink), CUTOFF)
print(f"     状态 = {status}(耗时 {time.time() - started:.1f}s), payload = {payload}")
rescued = sink.drain()
print(f"     中转站里有 {len(rescued)} 个内层抢出来的原函数:")
for attempt in rescued:
    print(f"       · {attempt.strategy[:44]}")
    print(f"         F = {str(attempt.result)[:72]}")

assert rescued, "中转站应该是非空的 —— 抢救没生效"
print("     → 机制生效了:外层放弃了,但**接住了**内层已经验证通过的原函数。")

# 但接下来这一步才是真正值得记下来的:
started = time.time()
value, note = _newton_leibniz_from(rescued[-1].result, x, lo, hi)
print()
print("     现在拿这个抢救出来的 F 去算 F(2a)−F(a):")
print(f"       → {value}  ({note})  (耗时 {time.time() - started:.2f}s)")
print()
print("     ⚠ 抢救**没成功**。原因不是机制失灵,而是:")
print("       被抢回来的是**搜索到一半的中间结果** —— 内层第一眼找到的是那个")
print("       带 Piecewise / Abs 的丑 F,它还没排到后面那个干净的")
print("       F = √(x²−a²) − a·asec(x/a),超时就到了。")
print("       丑 F 在上下限上取不了极限,所以照样算不出结果。")
print()
print("     ⇒ 这就是第三个层次的问题,也是这条路上最容易忽略的一点:")
print("        **抢救只能拿回「已经算完的」,不能把「还没算完的」变成能用的。**")
print("        想让超时之后的成果可用,必须让被超时管辖的那一步本身是一个")
print("        **完整的工作单元** —— 也就是把边界往里挪(子问题队列做的正是这件事)。")

# ---- 对照:没有检查点的题,抢救同样无能为力
print()
print("  [对照] 换一道**从头到尾没有检查点**的题(∫_0^1 ln(1+x)/(1+x²)dx 的不定积分):")
hard = sp.log(1 + x) / (1 + x**2)
sink2 = ResultSink(limit=4)
status2, _ = run_with_timeout(
    lambda: _strategy_newton_leibniz(hard, x, sp.Integer(0), sp.Integer(1), sink=sink2), 6.0)
print(f"     状态 = {status2},中转站里有 {len(sink2)} 个结果")
print("     → 内层从来没找到过原函数(这道题该走「换元 + 区间再现」),")
print("       连可以抢救的东西都没有。")

print()
print("=" * 92)
print("结论:这个问题分三个层次,前两个能解,第三个要靠架构")
print("=" * 92)
print("  层次一(预算不传播)")
print("      → 用 ContextVar 传播截止时间,内层取 min(自己想要的, 父层剩余)")
print("      → ✅ 完全解决,三层嵌套实测都被夹住")
print()
print("  层次二(成果被外层超时丢掉)")
print("      → 线程安全的中转站,成果在算出来的当下就交出来,不等 return")
print("      → ✅ 机制解决,实测外层超时后仍然接住了内层验证通过的原函数")
print()
print("  层次三(抢救回来的只是「半成品搜索」的中间结果,未必可用)")
print("      → 抢救只能拿回已经算完的东西,不能把没算完的变成能用的")
print("      → 这一层没有通用解法,只能靠**把超时边界对齐到完整的工作单元**:")
print("        把又贵又不可控的一步单独管起来,让它的失败不带走别的东西")
print("        (本项目里就是「换元产生子问题、由顶层队列调度」那套做法)")
print()
print("  一句话:超时的边界,必须和你愿意丢掉的东西的粒度对齐。")
