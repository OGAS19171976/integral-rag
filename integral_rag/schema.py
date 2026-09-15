"""方法卡片语料的 schema 定义与校验。

语料是这套系统的"知识",写错一个 trigger 就会导致检索把方法推荐错。
所以它的约束用代码固定下来,由 build_corpus.py 自动检查,而不是靠人肉校对。
"""

from __future__ import annotations

FAMILIES = {
    "direct_table",
    "substitution_basic",
    "substitution_algebraic",
    "substitution_trig",
    "parts",
    "partial_fractions",
    "rational_trig",
    "weierstrass",
    "euler_substitution",
    "reduction_formula",
    "special_trick",
}

# 四个触发字段各自允许出现的 token
TRIGGER_TOKENS = {
    "has_radical": {"a2-x2", "x2+a2", "x2-a2", "ax+b", "general", "none"},
    "has_func": {"poly", "exp", "log", "trig", "inv_trig", "radical",
                 "hyperbolic", "rational", "const", "abs"},
    "form": {"sum", "product", "quotient", "power", "composite", "rational",
             "rational_in_sin_cos", "any"},
    "misc": {"poly_times_exp", "poly_times_trig", "poly_times_log",
             "poly_times_invtrig", "odd_power_sin_cos", "even_power_sin_cos",
             "cyclic_parts", "improper_fraction", "exponential_rational",
             "repeated_root", "irreducible_quadratic", "binomial_power",
             # ---- 定积分/含参积分专用
             "oscillatory", "parameterized", "singular_endpoint",
             "singular_interior", "even_odd", "exponential_decay",
             "algebraic_decay", "abs_or_floor", "trig_orthogonality",
             "beta_shape", "gamma_shape", "frullani_shape",
             "power_denominator", "product_of_trig", "log_singularity"},
    # interval 只有定积分卡片会写
    "interval": {"symmetric", "zero_to_inf", "zero_to_pi_over_2",
                 "zero_to_pi", "zero_to_two_pi", "unit", "infinite",
                 "finite", "any"},
}

# 定积分方法卡片允许的 family
DEFINITE_FAMILIES = {
    "newton_leibniz", "symmetry", "king_property", "periodicity",
    "orthogonality", "wallis_reduction", "beta_gamma", "substitution_limits",
    "improper_split", "standard_result", "series_expansion", "parametric",
    "divergence_test",
}

DEFINITE_REQUIRED_FIELDS = ("id", "name", "family", "priority", "triggers",
                            "when", "recipe", "pitfalls", "examples")

REQUIRED_FIELDS = ("id", "name", "family", "priority", "triggers",
                   "when", "recipe", "pitfalls", "examples")

# 引擎里登记了代换生成器的卡片 —— 这些卡片能把建议变成可执行动作
ACTIONABLE_CARDS = {
    "trig_sub_a2_minus_x2", "trig_sub_x2_plus_a2", "trig_sub_x2_minus_a2",
    "hyperbolic_substitution", "radical_linear_substitution",
    "weierstrass_substitution", "denom_a_plus_b_trig",
    "exp_rational_substitution", "reciprocal_substitution",
    "sin_cos_odd_power", "tan_sec_powers", "euler_substitution",
    "algebraic_rationalization", "chebyshev_binomial",
}


def validate_card(card: dict, index: int = 0, definite: bool = False) -> list[str]:
    """返回这条卡片的所有问题(空列表表示没问题)。

    definite=True 时按定积分卡片的规则校验:family 取自 DEFINITE_FAMILIES,
    examples 必须是**完整定积分**(带上下限)。
    """
    errors: list[str] = []
    required = DEFINITE_REQUIRED_FIELDS if definite else REQUIRED_FIELDS
    allowed_families = DEFINITE_FAMILIES if definite else FAMILIES
    where = f"#{index}"

    if not isinstance(card, dict):
        return [f"{where} 不是 JSON 对象,而是 {type(card).__name__}"]

    where = f"#{index}({card.get('id', '?')})"

    for field in required:
        if field not in card:
            errors.append(f"{where} 缺少字段 {field}")

    if card.get("family") not in allowed_families:
        errors.append(f"{where} family 非法:{card.get('family')!r}")

    priority = card.get("priority")
    if not isinstance(priority, int) or isinstance(priority, bool):
        errors.append(f"{where} priority 必须是整数,实际是 {priority!r}")
    elif not 10 <= priority <= 90:
        errors.append(f"{where} priority={priority} 超出 10~90")

    triggers = card.get("triggers")
    if not isinstance(triggers, dict):
        errors.append(f"{where} triggers 必须是对象")
    else:
        for key, allowed in TRIGGER_TOKENS.items():
            if definite is False and key == "interval":
                # 不定积分卡片不该有 interval
                if key in triggers:
                    errors.append(f"{where} 不定积分卡片不应声明 triggers.interval")
                continue
            value = triggers.get(key)
            if value is None:
                continue
            if not isinstance(value, list):
                errors.append(f"{where} triggers.{key} 必须是数组")
                continue
            bad = [t for t in value if t not in allowed]
            if bad:
                errors.append(f"{where} triggers.{key} 含非法 token:{bad}")

    for field in ("when", "recipe", "pitfalls"):
        value = card.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{where} {field} 必须是非空字符串")

    examples = card.get("examples")
    if not isinstance(examples, list) or not examples:
        errors.append(f"{where} examples 必须是非空数组")
    else:
        for ex in examples:
            if not isinstance(ex, str) or not ex.strip():
                errors.append(f"{where} examples 含空项")
                continue
            has_limits = "\\int_" in ex or "\\int^" in ex
            if definite:
                if not has_limits:
                    errors.append(f"{where} 例题 {ex!r} 必须是带上下限的完整定积分")
                if not ex.strip().endswith("dx"):
                    errors.append(f"{where} 例题 {ex!r} 缺少 dx")
            else:
                if "\\int" in ex or ex.strip().endswith("dx"):
                    errors.append(f"{where} 例题 {ex!r} 应只含被积函数,不要 \\int 和 dx")

    return errors


def validate_corpus(cards: list, definite: bool = False) -> tuple[list[str], list[str]]:
    """返回 (错误列表, 警告列表)。"""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(cards, list):
        return ["语料顶层必须是 JSON 数组"], []

    seen: set[str] = set()
    for index, card in enumerate(cards):
        errors.extend(validate_card(card, index, definite=definite))
        if isinstance(card, dict):
            cid = card.get("id")
            if cid in seen:
                errors.append(f"id 重复:{cid}")
            seen.add(str(cid))

    expected_ids = _EXPECTED_DEFINITE_IDS if definite else _EXPECTED_IDS
    missing = {c for c in expected_ids if c not in seen}
    if missing:
        warnings.append(
            f"下面这些卡片 id 没有出现在语料里(检索覆盖面会变窄):\n    "
            + "\n    ".join(sorted(missing)))

    return errors, warnings


# 计划覆盖的 30 张卡片;缺哪张会在 warnings 里点名,但不当作错误
_EXPECTED_IDS = {
    "basic_table", "linear_substitution", "composite_substitution",
    "radical_linear_substitution", "trig_sub_a2_minus_x2",
    "trig_sub_x2_plus_a2", "trig_sub_x2_minus_a2", "reciprocal_substitution",
    "parts_poly_exp_trig", "parts_poly_log_invtrig", "parts_cyclic",
    "parts_tabular", "rational_distinct_linear", "rational_repeated_linear",
    "rational_irreducible_quadratic", "rational_improper_division",
    "weierstrass_substitution", "sin_cos_odd_power", "sin_cos_even_power",
    "trig_product_to_sum", "tan_sec_powers", "denom_a_plus_b_trig",
    "exp_rational_substitution", "chebyshev_binomial", "euler_substitution",
    "hyperbolic_substitution", "reduction_sin_cos_power",
    "reduction_quadratic_power", "conjugate_pairing", "algebraic_rationalization",
}

# 定积分计划覆盖的 27 张卡片
_EXPECTED_DEFINITE_IDS = {
    "def_newton_leibniz", "def_symmetry_even_odd", "def_king_property",
    "def_periodicity", "def_orthogonality", "def_wallis", "def_beta_gamma",
    "def_substitution_limits", "def_improper_split", "def_infinite_truncation",
    "def_standard_dirichlet", "def_standard_fresnel", "def_standard_poisson",
    "def_standard_power_denom", "def_standard_bose_fermi",
    "def_standard_frullani", "def_log_sine", "def_reciprocal_symmetry",
    "def_series_expansion", "def_param_continuity",
    "def_param_differentiation", "def_param_conditions",
    "def_param_uniform_convergence", "def_param_limit_interchange",
    "def_convergence_p_test", "def_convergence_comparison",
    "def_convergence_dirichlet_abel",
}
