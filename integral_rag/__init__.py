"""integral-rag —— 一元不定积分的「检索 + CAS + 验证」求解系统。

用法::

    from integral_rag import analyze

    report = analyze(r"\\int \\frac{x^2}{\\sqrt{1-x^2}} dx")
    print(report.render())
    print(report.result)

设计要点见 README.md。核心一句话:
    检索负责"想出方法",SymPy 负责"算出结果",验证负责"确认没错"。
"""

from .features import Features, extract, extract_definite, structural_similarity
from .parse import (IntegralInput, ParseError, parse_any_or_none, parse_integral,
                    parse_integrand, parse_limit, parse_or_none)
from .pipeline import (DefiniteReport, Report, analyze, analyze_definite,
                       find_method, get_definite_retriever, get_retriever,
                       solve_latex)
from .plan import Plan, make_plan
from .quadrature import NumericResult, evaluate
from .retrieve import Hit, Retriever
from .solve import Attempt, Solution, Substitution, solve, step_trace, ugliness
from .definite import DefiniteSolution, solve_definite
from .parametric import (DerivativeCheck, FeynmanResult, feynman,
                         verify_derivative_under_integral)
from .verify import VerifyResult, verify_antiderivative

__version__ = "0.2.0"

__all__ = [
    "analyze", "analyze_definite", "solve_latex", "find_method",
    "get_retriever", "get_definite_retriever", "Report", "DefiniteReport",
    "parse_integral", "parse_integrand", "parse_or_none", "parse_any_or_none",
    "parse_limit", "IntegralInput", "ParseError",
    "extract", "extract_definite", "Features", "structural_similarity",
    "Retriever", "Hit",
    "make_plan", "Plan",
    "solve", "Solution", "Substitution", "Attempt", "step_trace", "ugliness",
    "solve_definite", "DefiniteSolution",
    "evaluate", "NumericResult",
    "feynman", "FeynmanResult", "verify_derivative_under_integral", "DerivativeCheck",
    "verify_antiderivative", "VerifyResult",
]
