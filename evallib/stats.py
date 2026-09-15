"""评估用的统计层:把「点估计」升级成「带不确定性的结论」。

这个模块回答的只有一个问题:

    两个数字的差,是真的,还是抽样噪声?

它**刻意不依赖 scipy,也不依赖 integral_rag**,只用 numpy + 标准库:

1. 它要能原样搬进独立的 llm-eval-toolkit 仓库,不该拖着积分系统的依赖;
2. 它要能给别人用,少一个依赖就少一个装不上的理由;
3. 更实际的一点 —— 它要能在**没有网络、装不上新包**的环境里跑起来。
   (本机实测:pip 装包要跨沙箱授权,所以"零新依赖"不是洁癖,是可用性。)

设计上有个必须坚持的区分,贯穿本文件:

  * **重采样类**(Bootstrap / 置换检验):不假设分布,给区间和 p 值。
    适合"我不想假定指标是正态的"这种默认情形。
  * **解析类**(Wilson / 精确二项):样本小的时候比重采样更稳,而且可复现
    (Bootstrap 依赖随机种子,别人复现你的报告还得拿到种子)。

报告里两个都给,不一致的地方本身就是信息。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Iterable, Sequence

import numpy as np

_NORMAL = NormalDist()


# ======================================================================
# 一、比例的区间
# ======================================================================
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """二项比例的 Wilson 得分区间。

    为什么不用教科书那个 p̂ ± z√(p̂(1-p̂)/n):

      * p̂ = 0 或 1 时它的宽度是 **0** —— 56/56 会报出 [100%, 100%] 这种
        荒谬结果,而这恰恰是评估里最常见的情形(全对/全错);
      * 小样本下覆盖率明显偏低。

    本仓库现在的实测数据就是 56 题里 56 题全对。用 Wald 区间会得出
    "求解率 = 100% ± 0" 的结论,Wilson 给的是 [93.6%, 100%] —— 后者才是
    诚实的说法,也是面试里能被追问得住的说法。
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k 必须落在 [0, n] 内,收到 k={k}, n={n}")
    z = _NORMAL.inv_cdf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_interval_from_bools(flags: Iterable[bool], alpha: float = 0.05):
    flags = list(flags)
    return wilson_interval(sum(bool(f) for f in flags), len(flags), alpha)


# ======================================================================
# 二、Bootstrap
# ======================================================================
def _jackknife_acceleration(values: np.ndarray, stat: Callable) -> float:
    """BCa 需要的加速度 a:统计量对单个样本的敏感度(偏度)。

    分母为 0(统计量是常数)时返回 0,退化成 percentile 法。
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = stat(values)
    # 留一法:每次去掉第 i 个样本
    jack = np.array([stat(np.delete(values, i)) for i in range(n)], dtype=float)
    mean_jack = jack.mean()
    diffs = mean_jack - jack
    denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
    if denom == 0:
        return 0.0
    return float(np.sum(diffs ** 3) / denom)


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    stat: Callable = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    method: str = "bca",
    seed: int | None = 0,
) -> tuple[float, float]:
    """Bootstrap 置信区间。

    method:
      * "percentile" —— 直接取重采样分布的分位数,简单、好解释;
      * "bca" —— 偏差校正 + 加速度校正,小样本下覆盖率明显更好(默认)。

    seed 默认固定为 0:**报告里的数字必须能被他人在不看代码的情况下复现**。
    想要真正随机的重采样就显式传 seed=None。
    """
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        # 一个样本给不出区间。返回退化区间,并由上层决定要不要标成"不可估"。
        v = float(stat(arr))
        return (v, v)
    if method not in ("percentile", "bca"):
        raise ValueError(f"未知的 method: {method!r}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    if stat is np.mean:
        # 快路:评估里绝大多数统计量就是均值。向量化后快两个数量级,
        # 而下面那个通用回退分支保留给中位数、分位数这类无法向量化的统计量。
        boot = arr[idx].mean(axis=1)
    else:
        boot = np.array([stat(arr[i]) for i in idx], dtype=float)
    theta = float(stat(arr))

    if method == "percentile":
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))

    # ---- BCa ----
    prop = np.mean(boot < theta)
    # 全部重采样都 >= 或都 <= theta 时,Phi^{-1} 会发散,夹住避免 inf
    prop = min(max(prop, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = _NORMAL.inv_cdf(prop)
    acc = _jackknife_acceleration(arr, stat)

    def _adjust(z_alpha: float) -> float:
        denom = 1 - acc * (z0 + z_alpha)
        if denom == 0:
            return z_alpha
        return _NORMAL.cdf(z0 + (z0 + z_alpha) / denom)

    lo_p = _adjust(_NORMAL.inv_cdf(alpha / 2))
    hi_p = _adjust(_NORMAL.inv_cdf(1 - alpha / 2))
    lo, hi = np.percentile(boot, [100 * lo_p, 100 * hi_p])
    return (float(lo), float(hi))


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float]:
    """配对差异 mean(b) - mean(a) 的 Bootstrap 区间。

    **必须配对重采样**:同一个题目在 a、b 两个系统上的表现是相关的,
    分别独立重采样会把相关性丢掉,区间被系统性放大或缩小。
    """
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    if len(aa) != len(bb):
        raise ValueError(f"配对比较要求等长:len(a)={len(aa)}, len(b)={len(bb)}")
    diff = bb - aa
    return bootstrap_ci(diff, stat=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)


# ======================================================================
# 三、配对显著性检验
# ======================================================================
@dataclass
class McNemarResult:
    """同一批题、两个系统的配对比较结果。"""

    n_pairs: int
    both_pass: int          # a 对 b 也对
    a_only: int             # b 独有:改前错、改后对  (经典记号 c)
    b_only: int             # a 独有:改前对、改后错  (经典记号 b)
    both_fail: int
    exact_p: float          # 精确二项检验(推荐,小样本)
    mid_p: float            # mid-p 修正(不那么保守)
    chi2_p: float           # 连续性校正卡方近似(大样本,便于和文献对齐)
    rate_a: float
    rate_b: float

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only

    @property
    def delta(self) -> float:
        return self.rate_b - self.rate_a

    def render(self) -> str:
        lines = [
            f"配对样本 {self.n_pairs} 题   分歧对 {self.discordant} 题"
            f"(改后独对 {self.a_only} / 改后独错 {self.b_only})",
            f"  改前 {self.rate_a:.1%}  →  改后 {self.rate_b:.1%}"
            f"    Δ = {self.delta:+.1%}",
        ]
        if self.discordant == 0:
            lines.append("  两版**逐题完全相同**,任何检验都给不出差异 —— 差异为 0,不是「不显著」。")
            return "\n".join(lines)
        lines.append(
            f"  精确二项 p = {self.exact_p:.4f}   "
            f"mid-p = {self.mid_p:.4f}   "
            f"校正卡方 p = {self.chi2_p:.4f}"
        )
        return "\n".join(lines)


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k),X ~ Binomial(n, p)。n 小时用精确求和,大时用正态近似。"""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if n <= 1000:
        # 精确:math.comb 对 n <= 1000 完全够用,而且没有浮点累积误差。
        # p = 0.5 是 McNemar 唯一用到的情形,单独走整数路径(symmetry)。
        if p == 0.5:
            return sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
                   for i in range(0, k + 1))
    # 正态近似(连续性校正)
    mean = n * p
    sd = math.sqrt(n * p * (1 - p))
    return _NORMAL.cdf((k + 0.5 - mean) / sd)


def mcnemar(before: Sequence[bool], after: Sequence[bool]) -> McNemarResult:
    """McNemar 检验:同一批题上两个系统的配对比较。

    为什么评估必须用配对设计:同一道题在 A、B 两版上的难度是一样的,
    把这个"题目难度"配掉之后,要检出的差异只来自系统改动本身。
    把它当成两组独立样本来做卡方/比例检验,是把最贵的信息扔掉了。

    三个 p 值都给,不是凑数:

      * exact  —— 分歧对少于 ~25 时唯一可信的那个;
      * mid-p  —— 不那么保守,统计功效更高,现代推荐;
      * chi2   —— 大样本下和文献里的数字对齐,方便和别人比。

    它们**不一致本身**就是信息:分歧对很少的时候,卡方近似会失真。
    """
    b = [bool(v) for v in before]
    a = [bool(v) for v in after]
    if len(b) != len(a):
        raise ValueError(f"配对比较要求等长:len(before)={len(b)}, len(after)={len(a)}")
    if not b:
        raise ValueError("空评估集无法做配对检验")

    both_pass = sum(1 for x, y in zip(b, a) if x and y)
    b_only = sum(1 for x, y in zip(b, a) if x and not y)   # 改前对、改后错
    a_only = sum(1 for x, y in zip(b, a) if not x and y)   # 改前错、改后对
    both_fail = len(b) - both_pass - b_only - a_only

    n_disc = a_only + b_only
    rate_before = (both_pass + b_only) / len(b)
    rate_after = (both_pass + a_only) / len(b)

    if n_disc == 0:
        return McNemarResult(len(b), both_pass, a_only, b_only, both_fail,
                             1.0, 1.0, 1.0, rate_before, rate_after)

    m = min(a_only, b_only)
    p_less_eq = _binom_cdf(m, n_disc)
    exact_p = min(1.0, 2 * p_less_eq)
    # mid-p:把落在边界上的那一半概率还回去,少一点保守
    p_less = _binom_cdf(m - 1, n_disc)
    p_at = p_less_eq - p_less
    mid_p = min(1.0, 2 * (p_less + 0.5 * p_at))

    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
    chi2 = max(chi2, 0.0)
    # 1 自由度的卡方生存函数有闭式:p = erfc(sqrt(chi2/2))
    chi2_p = math.erfc(math.sqrt(chi2 / 2))

    return McNemarResult(len(b), both_pass, a_only, b_only, both_fail,
                         exact_p, mid_p, chi2_p, rate_before, rate_after)


def minimum_detectable_delta(
    n_pairs: int,
    alpha: float = 0.05,
) -> tuple[int, float]:
    """在 n_pairs 的配对评估里,净改善至少几道题(占多少)才算显著。

    返回 (最少净改善题数, 对应百分点)。

    做法:让改进全部集中在一个方向(改前错改后对 = k,反方向 = 0),找最小的
    k 使 McNemar 精确 p < alpha。这是**最好情形**下的门槛,所以是乐观下界 ——
    真实情况里分歧通常两个方向都有,门槛只会更高。

    **一个反直觉但很重要的推论**:McNemar 精确检验是以"分歧对的数量"为条件的,
    所以**总题量本身不提供额外判别力**。把评估集从 56 题扩到 200 题,只要新增
    的题两版表现都一样,检出能力**一点都不会提高**,纯粹白花钱。要提高判别力,
    只能有针对性地补那些"两版确实会分歧"的题 —— 这也是标注预算该往哪花。

    它的价值在于把"不显著"从一句死结论变成一句可执行的话:
    「56 题的规模下,少于 6 道题的净改善你根本测不出来。」
    """
    if n_pairs <= 0:
        return (-1, float("nan"))
    for k in range(1, n_pairs + 1):
        if min(1.0, 2 * _binom_cdf(0, k)) < alpha:
            return (k, k / n_pairs)
    return (-1, float("nan"))


def mcnemar_sample_size(
    p_discordant_a: float,
    p_discordant_b: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """要多少**配对样本**才能检出这个差异(Connor 1987 的闭式)。

    参数是两个方向的"分歧率":

      * p_discordant_a —— 改前对、改后错的比例(π01)
      * p_discordant_b —— 改前错、改后对的比例(π10)

    这个函数的存在本身就是回答面试题的:
    「要多少样本才能检出 5% 的提升?」——不算这个,你就只能回答"越多越好"。

    诚实说明:这是正态近似,且**对分歧率极其敏感**。分歧率未知时,
    先用一小批试点数据估出来,再代入 —— 拿它当"预算工具"而不是"承诺"。
    """
    if not 0 <= p_discordant_a < 1 or not 0 <= p_discordant_b < 1:
        raise ValueError("分歧率必须落在 [0, 1) 内")
    delta = p_discordant_b - p_discordant_a
    if delta == 0:
        return -1  # 差异为 0:再多样本也检不出
    psi = p_discordant_a + p_discordant_b
    z_a = _NORMAL.inv_cdf(1 - alpha / 2)
    z_b = _NORMAL.inv_cdf(power)
    inner = psi - delta * delta
    if inner < 0:
        inner = 0.0
    n = ((z_a * math.sqrt(psi) + z_b * math.sqrt(inner)) ** 2) / (delta * delta)
    return int(math.ceil(n))


def cochrans_q(conditions: Sequence[Sequence[bool]]) -> tuple[float, float]:
    """Cochran's Q:3 个及以上系统在同一批题上的整体差异检验。

    先做 Q,显著了再两两做 McNemar + 多重比较校正。
    顺序反了就是在钓鱼:直接跑 10 组两两比较,总有一组"显著"。
    """
    if len(conditions) < 2:
        raise ValueError("至少需要两个条件")
    mat = np.asarray([[bool(v) for v in c] for c in conditions], dtype=float)
    # mat 的行是**条件**、列是**题目**(Q 的公式按这个朝向写)
    k, n_items = mat.shape
    if k < 2:
        raise ValueError("至少需要两个条件")
    per_condition = mat.sum(axis=1)     # T_j
    per_item = mat.sum(axis=0)          # C_i
    total = float(mat.sum())
    if total == 0 or total == n_items * k:
        return 0.0, 1.0
    denom = k * total - float(np.sum(per_item ** 2))
    if denom == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * float(np.sum(per_condition ** 2)) - total ** 2) / denom
    q = max(q, 0.0)
    # 自由度 k-1 的卡方生存函数:df<=2 时有闭式,更大用 Wilson-Hilferty 近似
    df = k - 1
    if df == 1:
        p = math.erfc(math.sqrt(q / 2))
    elif df == 2:
        p = math.exp(-q / 2)
    else:
        z = ((q / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p = 1 - _NORMAL.cdf(z)
    return float(q), float(max(0.0, min(1.0, p)))


# ======================================================================
# 四、多重比较
# ======================================================================
def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05):
    """Holm 逐步下降法。返回 (调整后 p, 是否拒绝)。

    比 Bonferroni 一致地更不保守,且**不需要任何独立性假设** —— 评估里
    指标之间明显相关,VanderWaerden 之类的独立假设方法不能用。
    """
    ps = np.asarray(list(pvals), dtype=float)
    m = len(ps)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(ps)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * ps[idx]
        running = max(running, val)          # 保证单调不减
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05):
    """BH 法控制 FDR。返回 (调整后 p, 是否拒绝)。

    适用场景:你要同时报 6 个指标的显著性,而且**允许其中少数是假阳性**,
    只要总体假阳性比例受控。探索性分析用 BH,对外下结论用 Holm。
    """
    ps = np.asarray(list(pvals), dtype=float)
    m = len(ps)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(ps)
    adj = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = m / (rank + 1) * ps[idx]
        running = min(running, val)          # 从大到小,保证单调不增
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


# ======================================================================
# 五、评分者一致性
# ======================================================================
def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """两个评分者的一致性(去掉随机巧合之后)。

    评估里的用途很具体:**验证 LLM-as-Judge 能不能替代人工**。
    两个评委一致率 80% 听起来不错,但如果标签只有 2 类、且 90% 都是"通过",
    那么瞎猜也有一致率 —— Kappa 会把这份巧合扣掉。

    经验阈值(Landis & Koch):<0.20 差,0.21-0.40 一般,0.41-0.60 中等,
    0.61-0.80 较好,>0.80 很好。**报 Kappa 必须同时报一致率**,
    否则读者无法判断分歧发生在哪。
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("两个评分序列必须等长")
    if not a:
        raise ValueError("空序列")
    cats = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float:
    """多名评分者、每项评分人数相同的一致性。

    table[i][j] = 第 i 个题目被判为第 j 类的评委人数。
    比 Cohen's Kappa 多一个维度:评委可以是 3 个以上。
    """
    mat = np.asarray(table, dtype=float)
    if mat.ndim != 2:
        raise ValueError("table 必须是二维:行=题目,列=类别")
    n_items, n_cats = mat.shape
    if n_items == 0:
        raise ValueError("空表")
    counts = mat.sum(axis=1)
    if len(set(counts.tolist())) != 1:
        raise ValueError("Fleiss' Kappa 要求每项被评的评委人数相同")
    n_raters = int(counts[0])
    if n_raters < 2:
        raise ValueError("至少需要 2 名评委")
    p_i = (np.sum(mat ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_j = mat.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j ** 2))
    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_e) / (1 - p_e)


# ======================================================================
# 六、置信度校准
# ======================================================================
def reliability_curve(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    n_bins: int = 10,
) -> list[dict]:
    """校准曲线:模型说 80% 确定的时候,实际对了多少。"""
    conf = np.asarray(list(confidences), dtype=float)
    corr = np.asarray([bool(c) for c in corrects], dtype=float)
    if len(conf) != len(corr):
        raise ValueError("置信度与正确性序列必须等长")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            curve.append({"lo": float(lo), "hi": float(hi), "n": 0,
                          "confidence": None, "accuracy": None, "gap": None})
            continue
        avg_conf = float(conf[in_bin].mean())
        avg_acc = float(corr[in_bin].mean())
        curve.append({"lo": float(lo), "hi": float(hi), "n": n,
                      "confidence": avg_conf, "accuracy": avg_acc,
                      "gap": avg_acc - avg_conf})
    return curve


def expected_calibration_error(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """ECE:各分箱 |准确率 - 平均置信度| 的样本量加权平均。

    **ECE 必须和曲线一起报**,理由三条,都是这个数字本身藏起来的:

      * 它只给总量,不给**方向**。同样 ECE = 0.1 的两个系统,一个是全局
        轻微高估(调个温度标定就行),另一个是低端高估、高端低估(必须分段
        处理)—— 处置方式完全不同,而 ECE 把它们印成同一个数;
      * 它对**分箱数敏感**。n_bins 从 10 改成 15,ECE 通常会变,
        所以报 ECE 必须连 n_bins 一起报;
      * 分箱内样本少时它不稳定,而真实评估集常常只有一两百条。

    结论:ECE 适合**跨系统比较**,不适合单独作为"校准好不好"的结论。
    """
    n_total = len(list(confidences))
    if n_total == 0:
        return float("nan")
    curve = reliability_curve(confidences, corrects, n_bins)
    return float(sum(b["n"] / n_total * abs(b["gap"])
                     for b in curve if b["n"] > 0))


# ======================================================================
# 七、抽样设计
# ======================================================================
def stratified_sample(
    items: Sequence[dict],
    strata_key: str | Callable[[dict], str],
    n: int,
    seed: int | None = 0,
) -> list[dict]:
    """按层**等比例**抽样,层内随机。

    「100 道题」和「100 道按难度分层抽的题」不是一回事:前者很可能
    70% 是简单题,于是指标被简单题主导,而且**换一批题结论就变**。
    分层让每一层的权重固定下来,指标才有可复现的含义。

    这是纯标准库实现,不引 sklearn 的 train_test_split —— 那个分层
    在层样本量小于请求数时会静默少给,评估集规模对不上很难查。
    """
    items = list(items)
    if n >= len(items):
        return items
    keyfn = (lambda it: it.get(strata_key)) if isinstance(strata_key, str) else strata_key

    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(str(keyfn(it)), []).append(it)

    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    # 先按层大小等比例分配,再用最大余数法把零头补齐
    quotas: dict[str, float] = {g: n * len(v) / len(items) for g, v in groups.items()}
    base = {g: int(math.floor(q)) for g, q in quotas.items()}
    remainder = n - sum(base.values())
    for g, _ in sorted(quotas.items(), key=lambda kv: (-(kv[1] - math.floor(kv[1])), kv[0])):
        if remainder <= 0:
            break
        if base[g] < len(groups[g]):
            base[g] += 1
            remainder -= 1

    for g, k in base.items():
        pool = groups[g]
        if k <= 0:
            continue
        if k >= len(pool):
            picked.extend(pool)
            continue
        idx = rng.choice(len(pool), size=k, replace=False)
        picked.extend(pool[i] for i in sorted(idx.tolist()))
    return picked


def stratified_rate_ci(
    items: Sequence[dict],
    strata_key: str | Callable[[dict], str],
    flag_key: str,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float]]:
    """分层后的整体比例 + 区间。

    注意:**不要**对合并后的样本直接套 Wilson。合并比例是各层比例的加权和,
    它的方差不是二项方差 —— 直接套会把区间算错。这里按层算方差再按 w² 加权。

    诚实说明:这里给的是**正态近似区间,不是 Wilson**。分层估计量的分布
    没有 Wilson 那种闭式;样本小的时候它会偏窄(反保守)。当成"下界估计"
    用可以,别当成硬保证。层内样本量小于 ~10 时请直接看分层明细表。
    """
    items = list(items)
    if not items:
        return float("nan"), (float("nan"), float("nan"))
    keyfn = (lambda it: it.get(strata_key)) if isinstance(strata_key, str) else strata_key
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(str(keyfn(it)), []).append(it)

    total = len(items)
    p_hat = 0.0
    var = 0.0
    for g, pool in groups.items():
        w = len(pool) / total
        p_g = sum(bool(it.get(flag_key)) for it in pool) / len(pool)
        p_hat += w * p_g
        var += (w ** 2) * p_g * (1 - p_g) / len(pool)
    se = math.sqrt(var)
    z = _NORMAL.inv_cdf(1 - alpha / 2)
    return p_hat, (max(0.0, p_hat - z * se), min(1.0, p_hat + z * se))


# ======================================================================
# 八、报告里反复要用的组合
# ======================================================================
@dataclass
class RateEstimate:
    """一个比例的完整交代:点估计 + 三种区间 + 样本量。"""

    name: str
    k: int
    n: int
    wilson: tuple[float, float] = field(default=(float("nan"), float("nan")))
    bootstrap: tuple[float, float] = field(default=(float("nan"), float("nan")))

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def render(self) -> str:
        if self.n == 0:
            return f"{self.name:<28}  n=0  不可估"
        agree = "" if not (
            self.wilson[0] <= self.bootstrap[1] and self.bootstrap[0] <= self.wilson[1]
        ) else "  ⚠ 两法区间几乎不重叠,小样本警告"
        return (
            f"{self.name:<28} {self.k:>3}/{self.n:<3} = {self.rate:>6.1%}   "
            f"[Wilson {self.wilson[0]:.1%}, {self.wilson[1]:.1%}]   "
            f"[Bootstrap {self.bootstrap[0]:.1%}, {self.bootstrap[1]:.1%}]{agree}"
        )


def estimate_rate(
    name: str,
    flags: Sequence[bool],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> RateEstimate:
    """点的比例 + Wilson 区间 + Bootstrap 区间,一次给全。"""
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    if n == 0:
        return RateEstimate(name, 0, 0)
    as_float = np.array(flags, dtype=float)
    return RateEstimate(
        name=name, k=k, n=n,
        wilson=wilson_interval(k, n, alpha),
        bootstrap=bootstrap_ci(as_float, stat=np.mean, n_boot=n_boot,
                               alpha=alpha, seed=seed),
    )
