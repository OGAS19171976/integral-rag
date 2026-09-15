"""evallib.stats 的独立验证。

统计代码最危险的地方在于:**它错了也不会崩**,只会安静地给出一个看起来
很像样的数字。所以这里的策略是两条腿:

  1. **交叉验证** —— 用 scipy 当裁判。生产代码刻意零依赖(scipy 装不上时
     评估照样能跑),但测试里可以拿它来对答案。
  2. **手算** —— 手算得出来的小例子必须和实现一致(Cochran's Q、
     Fleiss' Kappa、ECE 这些没有现成裁判的,只能手推)。

运行:
    pytest tests/test_stats.py -q
    python tests/test_stats.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evallib import stats as S   # noqa: E402

try:
    import scipy.stats as sst
    HAVE_SCIPY = True
except ImportError:      # pragma: no cover - 本机有 scipy,这只是给别人的保护
    HAVE_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAVE_SCIPY, reason="需要 scipy 当裁判")


# ======================================================================
# Wilson
# ======================================================================
@needs_scipy
@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10),
                                 (56, 56), (0, 56), (55, 56), (3, 200), (197, 200)])
def test_wilson_matches_scipy(k, n):
    """Wilson 区间必须和 scipy 的实现逐位一致。"""
    mine = S.wilson_interval(k, n, alpha=0.05)
    ref = sst.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert mine[0] == pytest.approx(ref.low, abs=1e-12)
    assert mine[1] == pytest.approx(ref.high, abs=1e-12)


def test_wilson_all_pass_is_not_zero_width():
    """这条是本模块存在的理由之一。

    56 题全对,如果报 [100%, 100%] 就是把"没测出问题"说成了"没有问题"。
    Wilson 必须给出一个非退化的下界。
    """
    lo, hi = S.wilson_interval(56, 56)
    assert hi == pytest.approx(1.0)
    assert 0.93 < lo < 0.94, f"全对的 Wilson 下界应在 93.6% 附近,实得 {lo:.4f}"
    # 对照:教科书 Wald 区间在这里宽度为 0 —— 这正是不能用它的原因
    p = 1.0
    wald_half = 1.959963985 * math.sqrt(p * (1 - p) / 56)
    assert wald_half == 0.0


def test_wilson_all_fail_symmetric():
    lo, hi = S.wilson_interval(0, 56)
    assert lo == 0.0
    assert 0.06 < hi < 0.07, f"全错的 Wilson 上界应在 6.4% 附近,实得 {hi:.4f}"


def test_wilson_rejects_impossible_input():
    with pytest.raises(ValueError):
        S.wilson_interval(5, 3)


# ======================================================================
# McNemar
# ======================================================================
@needs_scipy
@pytest.mark.parametrize("a_only,b_only", [(10, 0), (0, 10), (10, 5), (30, 20),
                                           (5, 5), (1, 0), (25, 24), (100, 80)])
def test_mcnemar_exact_matches_scipy(a_only, b_only):
    """精确二项 p 必须和 scipy 的两侧精确检验一致。"""
    before = [False] * a_only + [True] * b_only
    after = [True] * a_only + [False] * b_only
    res = S.mcnemar(before, after)
    n_disc = a_only + b_only
    m = min(a_only, b_only)
    ref = sst.binomtest(m, n_disc, 0.5, alternative="two-sided").pvalue
    assert res.exact_p == pytest.approx(ref, abs=1e-12)


@needs_scipy
@pytest.mark.parametrize("a_only,b_only", [(10, 0), (10, 5), (30, 20), (100, 80)])
def test_mcnemar_chi2_matches_scipy(a_only, b_only):
    """连续性校正卡方要和 scipy 的卡方生存函数一致。"""
    before = [False] * a_only + [True] * b_only
    after = [True] * a_only + [False] * b_only
    res = S.mcnemar(before, after)
    chi2 = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    ref = sst.chi2.sf(chi2, 1)
    assert res.chi2_p == pytest.approx(ref, abs=1e-12)


def test_mcnemar_hand_computed():
    """手算:b=0, c=10 时精确 p = 2 * (1/2)^10 = 1/512。"""
    res = S.mcnemar([True] * 10, [False] * 10)
    assert res.b_only == 10 and res.a_only == 0
    assert res.exact_p == pytest.approx(1 / 512, abs=1e-12)
    assert res.rate_a == 1.0 and res.rate_b == 0.0
    assert res.delta == pytest.approx(-1.0)


def test_mcnemar_identical_versions_is_not_significant():
    """两版逐题完全相同时:分歧对为 0,p = 1,而且方向必须是"无差异"而不是"不显著"。"""
    flags = [True, False, True, True, False]
    res = S.mcnemar(flags, list(flags))
    assert res.discordant == 0
    assert res.exact_p == 1.0
    assert res.delta == 0.0
    assert "完全相同" in res.render()


def test_mcnemar_mid_p_is_never_larger_than_exact():
    """mid-p 的设计目的就是"少保守一点",它必须 <= 精确 p,且 >= 精确 p 的一半。"""
    for a_only, b_only in [(1, 0), (3, 1), (10, 5), (25, 20), (40, 10)]:
        before = [False] * a_only + [True] * b_only
        after = [True] * a_only + [False] * b_only
        res = S.mcnemar(before, after)
        assert res.mid_p <= res.exact_p + 1e-12
        assert res.mid_p >= res.exact_p / 2 - 1e-12


def test_mcnemar_rejects_unpaired():
    with pytest.raises(ValueError):
        S.mcnemar([True, False], [True])
    with pytest.raises(ValueError):
        S.mcnemar([], [])


def test_mcnemar_sample_size_monotone_and_sane():
    """效应越大、要求功效越高,需要的样本越多 —— 以及小效应的量级要合理。"""
    small = S.mcnemar_sample_size(0.10, 0.15)   # Δ=5%
    big = S.mcnemar_sample_size(0.10, 0.30)     # Δ=20%
    assert big < small
    assert small > 500, f"检出 5% 差异不该少于几百题,实得 {small}"
    assert S.mcnemar_sample_size(0.10, 0.40) < big
    # 功效更高 → 需要更多样本
    assert S.mcnemar_sample_size(0.10, 0.15, power=0.9) > small
    # 差异为 0:再多样本也检不出,返回 -1 而不是一个假的有限值
    assert S.mcnemar_sample_size(0.10, 0.10) == -1


# ======================================================================
# 多重比较
# ======================================================================
@needs_scipy
def test_bh_matches_scipy():
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    adj, rej = S.benjamini_hochberg(ps, alpha=0.05)
    ref = sst.false_discovery_control(ps)
    assert list(adj) == pytest.approx(list(ref), abs=1e-12)


def test_holm_hand_computed():
    """手算验证:样本量与上例相同。"""
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    expected = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * ps[idx])
        expected[idx] = min(1.0, running)
    adj, rej = S.holm_bonferroni(ps, alpha=0.05)
    assert list(adj) == pytest.approx(expected, abs=1e-12)
    # Holm 恒不弱于 Bonferroni(逐项 <=),也比它更不保守
    assert all(a <= min(1.0, m * p) + 1e-12 for a, p in zip(adj, ps))


def test_adjustments_preserve_p_value_order():
    """保序性:原始 p 更小 ⇒ 调整后 p 不会反而更大。

    (不能用"排序后的下标序列相等"来断言:调整后会出现大量并列的 1.0,
    并列时的先后顺序没有意义,那样写是在测 Python 的排序稳定性。)
    """
    ps = [0.9, 0.01, 0.5, 0.03, 0.2]
    for fn in (S.holm_bonferroni, S.benjamini_hochberg):
        adj, _ = fn(ps, alpha=0.05)
        assert all(0.0 <= a <= 1.0 for a in adj)
        for i in range(len(ps)):
            for j in range(len(ps)):
                if ps[i] < ps[j]:
                    assert adj[i] <= adj[j] + 1e-12


def test_multiple_comparison_control_actually_controls():
    """不打校正 vs 打校正:同一批边缘显著的 p 值,结论必须被收紧。"""
    ps = [0.049, 0.048, 0.047, 0.046, 0.045, 0.044, 0.043, 0.042]
    naive = [p < 0.05 for p in ps]
    assert sum(naive) == len(ps)          # 裸看全都"显著"
    _, holm = S.holm_bonferroni(ps, 0.05)
    assert sum(holm) == 0                 # 校正后一个都不显著


# ======================================================================
# 一致性
# ======================================================================
def test_cohen_kappa_perfect_and_chance():
    assert S.cohen_kappa([1, 2, 1, 2], [1, 2, 1, 2]) == pytest.approx(1.0)
    # 完全反向:一致率 0,Kappa 为负
    assert S.cohen_kappa([1, 1, 2, 2], [2, 2, 1, 1]) < 0
    # 手算:一致 8/10,两边边缘分布都是 5:5 → pe = 0.5 → kappa = (0.8-0.5)/0.5 = 0.6
    a = [1] * 5 + [2] * 5
    b = [1] * 4 + [2] * 5 + [1]
    assert sum(x == y for x, y in zip(a, b)) == 8
    assert S.cohen_kappa(a, b) == pytest.approx(0.6)


def test_kappa_exposes_the_high_agreement_trap():
    """这个测试说明**为什么不能只报一致率**。

    a、b 的边缘分布完全一样(95% 通过),逐题却几乎对不上:
    原始一致率 90% 看着很好,扣掉巧合之后 Kappa 是负的 —— 比瞎猜还差。
    评估里"两边都判通过"占绝大多数时,只报一致率一定会骗人。
    """
    a = [1] * 95 + [2] * 5
    b = [1] * 90 + [2] * 5 + [1] * 5
    raw = sum(x == y for x, y in zip(a, b)) / len(a)
    kappa = S.cohen_kappa(a, b)
    assert raw == pytest.approx(0.90), f"原始一致率 {raw:.2f}"
    assert kappa < 0.05, f"扣掉巧合后应接近 0 甚至为负,实得 {kappa:.3f}"


def test_fleiss_kappa_hand_computed():
    """手算:3 项、2 类、2 评委。
       item1 [2,0], item2 [1,1], item3 [0,2]
       P_bar = (1 + 0 + 1)/3 = 2/3,Pe = 0.5 → kappa = 1/3
    """
    assert S.fleiss_kappa([[2, 0], [1, 1], [0, 2]]) == pytest.approx(1 / 3)
    assert S.fleiss_kappa([[3, 0], [0, 3]]) == pytest.approx(1.0)


def test_fleiss_kappa_rejects_unequal_raters():
    """每项被评的评委数不等时,Fleiss' Kappa 没有定义,必须报错而不是硬算。"""
    with pytest.raises(ValueError):
        S.fleiss_kappa([[2, 0], [1, 0]])


def test_fleiss_kappa_realistic_three_raters():
    """3 名评委、4 项、3 类。逐项 P_i 与 Kappa 全部手算。"""
    table = [[3, 0, 0],
             [0, 2, 1],
             [1, 1, 1],
             [0, 0, 3]]
    n = 3
    p_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in table]
    assert p_i == pytest.approx([1.0, 1 / 3, 0.0, 1.0])
    p_bar = sum(p_i) / 4
    col = [sum(table[r][c] for r in range(4)) / 12 for c in range(3)]
    p_e = sum(x * x for x in col)
    assert S.fleiss_kappa(table) == pytest.approx((p_bar - p_e) / (1 - p_e))


# ======================================================================
# 校准
# ======================================================================
def test_ece_hand_computed():
    """手算:5 个样本都落在 (0.8, 0.9] 箱,平均置信度 0.9、实际准确率 0.8 → ECE = 0.1。"""
    ece = S.expected_calibration_error([0.9] * 5, [True] * 4 + [False], n_bins=10)
    assert ece == pytest.approx(0.1, abs=1e-12)


def test_ece_perfect_calibration_is_zero():
    """置信度逐箱等于实际准确率时,ECE 必须精确为 0(不是"大概很小")。"""
    conf, corr = [], []
    for c in (0.0, 0.1, 0.9, 1.0):
        conf += [c] * 10
        corr += [True] * round(c * 10) + [False] * (10 - round(c * 10))
    ece = S.expected_calibration_error(conf, corr, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_nonzero_ece_cannot_cancel_across_bins():
    """写这个测试是为了纠正一个常见误解:ECE 用的是 |gap|,**不会互相抵消**。

    两端符号相反时 ECE 只会更大,不会变小。真正的风险是另一回事 ——
    见下一个测试。
    """
    conf = [0.1] * 10 + [0.9] * 90
    corr = [True] * 10 + [False] * 90
    curve = S.reliability_curve(conf, corr, n_bins=10)
    used = [b for b in curve if b["n"] > 0]
    gaps = sorted(b["gap"] for b in used)
    assert gaps[0] < 0 and gaps[1] > 0, "两端符号相反"
    ece = S.expected_calibration_error(conf, corr, n_bins=10)
    # 平均 |gap| 是 0.9,加权后仍是 0.9 —— 没有抵消,只有放大
    assert ece == pytest.approx(0.9, abs=1e-12)


def test_same_ece_two_completely_different_problems():
    """这条才是"必须同时报曲线"的真正理由。

    两个系统的 ECE 都是 0.1,但:
      A 全局一致地轻微高估  → 调一个温度参数就能修好;
      B 低端高估、高端低估  → 必须分段校正,调温度只会把一头弄得更糟。
    ECE 把这两种情况印成同一个数字,只看它就会做出错误的处置决定。
    """
    conf_a = [0.6] * 100
    corr_a = [True] * 50 + [False] * 50          # 准确率恒为 0.5

    conf_b = [0.1] * 50 + [0.9] * 50
    corr_b = [False] * 50 + [True] * 50          # 低端 0% 命中、高端 100% 命中

    ece_a = S.expected_calibration_error(conf_a, corr_a, n_bins=10)
    ece_b = S.expected_calibration_error(conf_b, corr_b, n_bins=10)
    assert ece_a == pytest.approx(0.1, abs=1e-12)
    assert ece_b == pytest.approx(0.1, abs=1e-12)
    assert ece_a == pytest.approx(ece_b), "ECE 完全相同 —— 这正是它不够用的地方"

    gap_a = [b["gap"] for b in S.reliability_curve(conf_a, corr_a, n_bins=10)
             if b["n"] > 0]
    gap_b = [b["gap"] for b in S.reliability_curve(conf_b, corr_b, n_bins=10)
             if b["n"] > 0]
    assert len(gap_a) == 1 and gap_a[0] < 0, "A 只有单一方向"
    assert min(gap_b) < 0 < max(gap_b), "B 两端方向相反 —— 曲线才能看出来"
    assert len(gap_b) == 2


# ======================================================================
# Bootstrap
# ======================================================================
def test_bootstrap_is_reproducible_with_seed():
    """同一份数据、同一个种子,两次结果必须完全一致 —— 报告要能被别人复现。"""
    rng = np.random.default_rng(7)
    data = (rng.random(80) < 0.4).astype(float)
    a = S.bootstrap_ci(data, seed=123)
    b = S.bootstrap_ci(data, seed=123)
    assert a == b
    # percentile 与 bca 两种方法都要能给出合法区间
    lo, hi = S.bootstrap_ci(data, seed=123, method="percentile")
    assert lo <= hi


def test_bootstrap_seed_actually_changes_the_draw():
    """种子必须真的影响重采样。

    注意这里用了**连续**统计量,这不是随手选的。0/1 数据的 Bootstrap 均值
    只能取 k/n 这些离散值,1 万次重采样下分位数会稳定地落在同一对次序统计量
    上,于是**不同种子给出完全相同的区间**。那是数据分辨率的问题,不是种子
    没生效 —— 拿离散数据断言"不同种子必然不同"会得到一个假警报。
    (这条是实测撞出来的:一开始就是用 0/1 数据写的,结果两个种子给出
     一模一样的 (0.2875, 0.5)。)
    """
    rng = np.random.default_rng(7)
    continuous = rng.normal(size=60)
    assert S.bootstrap_ci(continuous, seed=123) != S.bootstrap_ci(continuous, seed=124)


def test_bootstrap_seed_none_is_still_valid():
    rng = np.random.default_rng(7)
    data = rng.normal(size=50)
    lo, hi = S.bootstrap_ci(data, seed=None)
    assert lo < hi


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    data = (rng.random(200) < 0.35).astype(float)
    lo, hi = S.bootstrap_ci(data, method="percentile")
    assert lo < data.mean() < hi
    lo_b, hi_b = S.bootstrap_ci(data, method="bca")
    assert lo_b < data.mean() < hi_b


def test_bootstrap_ci_shrinks_with_more_data():
    rng = np.random.default_rng(11)
    small = (rng.random(40) < 0.5).astype(float)
    large = (rng.random(4000) < 0.5).astype(float)
    w_small = np.subtract(*reversed(S.bootstrap_ci(small, method="percentile")))
    w_large = np.subtract(*reversed(S.bootstrap_ci(large, method="percentile")))
    assert w_large < w_small / 3


def test_bootstrap_coverage_is_close_to_nominal():
    """覆盖率的实证检查:名义 95% 的区间,真实覆盖率不该掉到 88% 以下。

    这是唯一能抓住"公式抄错了但看起来正常"的测试 —— 逐位比对能验公式,
    覆盖率能验**语义**。
    """
    rng = np.random.default_rng(2024)
    p_true, n, reps = 0.30, 60, 200
    covered = 0
    for r in range(reps):
        data = (rng.random(n) < p_true).astype(float)
        lo, hi = S.bootstrap_ci(data, n_boot=800, seed=r)
        covered += int(lo <= p_true <= hi)
    coverage = covered / reps
    assert coverage >= 0.88, f"95% 区间的实测覆盖率只有 {coverage:.1%}"
    assert coverage <= 1.0


def test_wilson_coverage_beats_wald_on_extreme_rates():
    """p 接近 0、样本不大时,Wald 区间的覆盖率会崩,Wilson 不会。

    这条把"为什么本模块坚持用 Wilson"从观点变成了可执行的证据。
    """
    rng = np.random.default_rng(99)
    p_true, n, reps = 0.02, 50, 400
    cov_wilson = cov_wald = 0
    z = 1.959963985
    for r in range(reps):
        data = (rng.random(n) < p_true).astype(float)
        k = int(data.sum())
        lo, hi = S.wilson_interval(k, n)
        cov_wilson += int(lo <= p_true <= hi)
        p_hat = k / n
        half = z * math.sqrt(p_hat * (1 - p_hat) / n)
        cov_wald += int(p_hat - half <= p_true <= p_hat + half)
    assert cov_wilson / reps > cov_wald / reps, \
        f"Wilson {cov_wilson/reps:.1%} 应优于 Wald {cov_wald/reps:.1%}"
    assert cov_wilson / reps >= 0.90


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        S.paired_bootstrap_diff_ci([1, 0, 1], [1, 0])


# ======================================================================
# 抽样
# ======================================================================
def test_stratified_sample_respects_proportions():
    """层配额必须等比例,且总量精确等于 n(最大余数法要补齐零头)。"""
    items = [{"id": i, "level": ("易" if i < 60 else "难" if i < 90 else "极难")}
             for i in range(100)]
    picked = S.stratified_sample(items, "level", n=20, seed=0)
    assert len(picked) == 20
    got = {}
    for it in picked:
        got[it["level"]] = got.get(it["level"], 0) + 1
    assert got == {"易": 12, "难": 6, "极难": 2}


def test_stratified_sample_is_deterministic_and_without_duplicates():
    items = [{"id": i, "g": i % 3} for i in range(60)]
    a = S.stratified_sample(items, "g", 15, seed=5)
    b = S.stratified_sample(items, "g", 15, seed=5)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert len({x["id"] for x in a}) == 15


def test_stratified_sample_passthrough_when_n_covers_all():
    items = [{"id": i, "g": i % 2} for i in range(8)]
    assert S.stratified_sample(items, "g", 8) == items
    assert len(S.stratified_sample(items, "g", 99)) == 8


# ======================================================================
# Cochran's Q
# ======================================================================
@needs_scipy
def test_cochrans_q_matches_scipy():
    conds = [[True, True, False, True, False, False, True, False],
             [True, False, False, True, False, False, True, True],
             [False, False, False, True, True, False, True, True]]
    q, p = S.cochrans_q(conds)
    from scipy.stats import chi2
    # 手算 Q,再用 scipy 的卡方生存函数当裁判
    import numpy as np
    m = np.asarray(conds, dtype=float)
    k, n_items = m.shape
    total = m.sum()
    col = m.sum(axis=1)
    row = m.sum(axis=0)
    q_ref = (k - 1) * (k * (col ** 2).sum() - total ** 2) / (k * total - (row ** 2).sum())
    assert q == pytest.approx(q_ref, abs=1e-12)
    assert p == pytest.approx(chi2.sf(q_ref, k - 1), abs=1e-12)


def test_cochrans_q_no_difference():
    same = [True, False, True, True]
    q, p = S.cochrans_q([same, same, same])
    assert q == 0.0 and p == 1.0


# ======================================================================
# RateEstimate
# ======================================================================
def test_estimate_rate_render():
    est = S.estimate_rate("求解率", [True] * 56)
    assert est.k == 56 and est.n == 56 and est.rate == 1.0
    assert est.wilson[1] == 1.0 and est.wilson[0] < 1.0
    text = est.render()
    assert "56/56" in text and "100.0%" in text and "Wilson" in text


def test_estimate_rate_empty():
    est = S.estimate_rate("空", [])
    assert math.isnan(est.rate)
    assert "不可估" in est.render()


def test_wilson_interval_from_bools():
    assert S.wilson_interval_from_bools([True, True, False, False]) == \
           S.wilson_interval(2, 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
