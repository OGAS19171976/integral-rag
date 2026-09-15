"""LaTeX / 纯文本 输入 -> SymPy 表达式。

为什么不用 sympy.parsing.latex.parse_latex?
    它依赖 antlr4 运行时,本机没装;而且报错信息几乎不可读。
    一元不定积分用到的 LaTeX 语法非常有限,自己写一个转换器,
    每条替换规则都可见、可调试 —— 出问题时能直接看到中间串。

这个转换器比"最小实现"多做了两件必须要做的事:

  1. 显式捕获函数参数。LaTeX 里 ``\\sin x``、``\\sin(x)``、``\\sin{x}``
     三种写法等价,而且 ``\\sin 3x`` 的参数是 ``3x`` 不是 ``3``。
     靠 SymPy 的 implicit_application 去猜会踩坑(它可能给出 sin(3)*x),
     所以这里自己读:参数一直延伸到下一个顶层 ``\\`` 命令、``+``、``-`` 或结束。

  2. 处理 LaTeX 的幂记号 ``\\sin^{3}x``。它的含义是 (sin x)³,
     而不是 sin 的 3 次幂。这是三角函数幂次题的标准写法,
     不处理的话 ``\\sin^{3}x`` 会被解析成 ``sin**(3)*x`` 这种无意义的式子。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    NAME,
    auto_symbol,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------- 符号表
# 默认假设:积分变量取实数,参数取正数。
# 数学分析教材里 a、b、n 基本都是正的,给出这个假设能让 SymPy 少产生
# 一堆 sqrt(a**2)=Abs(a) 形式的中间结果。这属于「教学场景的合理约定」,
# 遇到需要 a<0 的题(极少)请显式代入具体数字。
SYMBOLS: dict[str, sp.Symbol] = {}
for _name in "abckmnpqrs":
    SYMBOLS[_name] = sp.Symbol(_name, positive=True)
for _name in "n":
    SYMBOLS[_name] = sp.Symbol(_name, positive=True, integer=True)
for _name in "xyzuvwt":
    SYMBOLS[_name] = sp.Symbol(_name, real=True)

DEFAULT_VAR = SYMBOLS["x"]


# ---------------------------------------------------------------- 未知标识符
# SymPy 自带的 auto_symbol 会给未声明的名字创建**没有任何假设**的符号,
# 于是 `sym.is_real` 是 None(而不是 True)。这会静静地毁掉含参积分:
# 参数识别用的是 `is_real`,一旦为 None 参数就被当成不存在,
# 整个"把参数代进数值层核对"的机制直接失效
# (实测 ∫_0^∞ x^{s-1}e^{-x}dx 里 s 不在符号表,答案 gamma(s) 摆在那儿却验不了)。
#
# 注意**不要**试图替换 auto_symbol 本身。它产生的是 Symbol **对象** token,
# 后续的 implicit_multiplication 等变换依赖那一套 token 约定;
# 自己写一个同样产生 Symbol token 的变换,最后会在 untokenize 处炸成
# "unsupported operand type(s) for +=: 'Symbol' and 'str'"。
# 稳妥的做法是照常解析,解析完再给无假设的符号补上 real=True。
def _assume_real(expr: sp.Expr) -> sp.Expr:
    """给「没有任何假设」的自由符号补上 real=True。"""
    replacements = {}
    for sym in expr.free_symbols:
        if sym.is_real is None:
            replacements[sym] = sp.Symbol(sym.name, real=True)
    if not replacements:
        return expr
    try:
        return expr.xreplace(replacements)
    except Exception:
        return expr


_TRANSFORMS = standard_transformations

# LaTeX 命令 -> SymPy 函数名。\ln / \lg 统一映射到 log(即自然对数),
# 这是国内教材的写法惯例。
_FUNC_MAP = {
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot",
    "sec": "sec", "csc": "csc",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "ln": "log", "lg": "log", "log": "log",
    "exp": "exp",
}
# 必须按长度降序匹配,否则 \sinh 会被 \sin 抢先吃掉
_FUNC_NAMES = tuple(sorted(_FUNC_MAP, key=len, reverse=True))


class ParseError(ValueError):
    """输入无法转换为 SymPy 表达式。"""


# parse_expr 内部走的是 eval。输入来自使用者自己,但仍顺手挡掉明显的
# 代码注入写法 —— 成本几乎为零。
_FORBIDDEN = re.compile(
    r"__|;|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|\bsys\b|\bglobals\b|\blocals\b")


def _guard(raw: str) -> None:
    hit = _FORBIDDEN.search(raw)
    if hit:
        raise ParseError(f"输入里出现了不被允许的内容 {hit.group(0)!r},已拒绝解析")


# ---------------------------------------------------------------- 花括号
def _read_balanced(s: str, i: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """从 s[i] == open_ch 开始读取配对的分组,返回 (内容, 结束下标)。"""
    if i >= len(s) or s[i] != open_ch:
        raise ParseError(f"期望 {open_ch!r},实际是 {s[i:i + 10]!r}")
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_ch:
            depth += 1
        elif s[j] == close_ch:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ParseError(f"{open_ch} 不配对")


def _push(out: list[str], text: str) -> None:
    """把展开结果追加到输出,必要时补一个乘号。

    没有这一步的话,x\\sqrt{1+x²} 会先变成 "xsqrt(...)" —— 两个名字粘在
    一起被当成一个符号名,然后被拆成 q*r*s*t*x 这种莫名其妙的东西。
    补上 'x*sqrt(...)' 就干净了。
    """
    if out:
        prev = out[-1]
        if prev and (prev[-1].isalnum() or prev[-1] in ")}]"):
            out.append("*")
    out.append(text)


def _expand_frac(s: str) -> str:
    """把 \\frac{A}{B} 替换成 ((A)/(B))。"""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith("\\frac", i):
            j = i + 5
            if j < len(s) and s[j] == "{":
                num, j2 = _read_balanced(s, j, "{", "}")
                if j2 < len(s) and s[j2] == "{":
                    den, j3 = _read_balanced(s, j2, "{", "}")
                    # 有些教材把微分写在分子里,写成 \frac{dx}{x^{2}}。
                    # 微分被 _strip_integral 摘掉之后分子就空了,补成 1。
                    if not num.strip():
                        num = "1"
                    _push(out, f"(({num})/({den}))")
                    i = j3
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _expand_sqrt(s: str) -> str:
    """\\sqrt{A} -> sqrt(A);\\sqrt[n]{A} -> ((A)**(1/(n)))。"""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith("\\sqrt", i):
            j = i + 5
            root = None
            if j < len(s) and s[j] == "[":
                k = s.find("]", j)
                if k == -1:
                    raise ParseError("\\sqrt[ 没有闭合的 ]")
                root = s[j + 1:k]
                j = k + 1
            if j < len(s) and s[j] == "{":
                inner, j2 = _read_balanced(s, j, "{", "}")
                _push(out, f"(({inner})**(1/({root})))" if root else f"sqrt({inner})")
                i = j2
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


# ---------------------------------------------------------------- 函数参数
# 裸参数(没有花括号/圆括号)的停止符:下一个 LaTeX 命令、加减号、逗号,
# 以及所属分组的右括号。这就是 LaTeX 的阅读约定:\sin 3x\cos 5x
# 的读者理解是 sin(3x)·cos(5x),不是 sin·3·x·cos·5·x。
_BARE_STOP = set("\\+-=,*/")


def _read_exponent(s: str, i: int) -> tuple[str | None, int]:
    """读取 ^{...} / ^n / ^-1。i 指向函数名之后。"""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s) or s[i] != "^":
        return None, i
    i += 1
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "{":
        return _read_balanced(s, i, "{", "}")
    start = i
    if i < len(s) and s[i] == "-":
        i += 1
    while i < len(s) and (s[i].isalnum() or s[i] == "."):
        i += 1
    if start == i:
        return None, start
    # 注意:\sin^{-1}x 在这里被解释成 1/sin x,而不是 arcsin x。
    # 后者是某些教材的写法,属于歧义,遇到请直接写 \arcsin x。
    return s[start:i], i


def _read_arg(s: str, i: int) -> tuple[str, int]:
    """读取一个函数参数。返回 (参数原文, 下一个位置)。

    参数可能是花括号组、圆括号组、裸的单项式(如 3x),或者另一个 LaTeX
    命令(如 \\sin\\frac{1}{x})。返回的是原文,展开交给递归调用去做。
    """
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise ParseError("函数缺少参数")
    if s[i] == "{":
        return _read_balanced(s, i, "{", "}")
    if s[i] == "(":
        return _read_balanced(s, i, "(", ")")

    if s[i] == "\\":
        # 参数本身是一个 LaTeX 命令
        for name in _FUNC_NAMES:
            if s.startswith(name, i + 1):
                j = i + 1 + len(name)
                try:
                    _, j = _read_exponent(s, j)
                    _, j = _read_arg(s, j)
                except ParseError:
                    return s[i:i + 1 + len(name)], i + 1 + len(name)
                return s[i:j], j
        match = re.match(r"\\([A-Za-z]+)", s[i:])
        if match:
            j = i + match.end()
            while j < len(s) and s[j] == "{":
                _, j = _read_balanced(s, j, "{", "}")
            return s[i:j], j

    start = i
    depth = 0
    while i < len(s):
        ch = s[i]
        if ch in "({":
            depth += 1
        elif ch in ")}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and ch in _BARE_STOP:
            break
        i += 1
    arg = s[start:i]
    if not arg.strip():
        raise ParseError("函数缺少参数")
    return arg, i


def _expand_abs(s: str) -> str:
    """把 |...| 换成 Abs(...)。

    LaTeX 里绝对值就是这么写的:`|\\sin x|`、`\\left|x\\right|`。
    不做这一步的话,`|` 会被 Python 解析器当成按位或,直接语法错误。
    用栈配对,所以嵌套的 ||x|-1| 也能处理。
    """
    out: list[str] = []
    stack: list[int] = []
    for ch in s:
        if ch == "|":
            if stack:
                start = stack.pop()
                out[start] = "Abs("
                out.append(")")
            else:
                stack.append(len(out))
                out.append("|")          # 占位,配对时替换成 Abs(
        else:
            out.append(ch)
    return "".join(out)


def _expand_functions(s: str) -> str:
    """把 \\sin x / \\sin(x) / \\sin{x} / \\sin^{3}x 统一成 SymPy 写法。

    对参数递归调用,所以 \\sqrt{\\sin x} 这种嵌套也能处理。
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "\\":
            matched = None
            for name in _FUNC_NAMES:
                if s.startswith(name, i + 1):
                    matched = name
                    break
            if matched:
                j = i + 1 + len(matched)
                try:
                    exponent, j = _read_exponent(s, j)
                    arg, j = _read_arg(s, j)
                except ParseError:
                    matched = None
                if matched:
                    call = f"{_FUNC_MAP[matched]}({_expand_functions(arg)})"
                    if exponent is not None:
                        call = f"({call})**({exponent})"
                    _push(out, call)
                    i = j
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


# ---------------------------------------------------------------- 清理
_NOISE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|limits|nolimits"
    r"|biggl|biggr|bigl|bigr|Bigl|Bigr|bigg|Bigg|big|Big"
    r"|left|right|quad|qquad|,|!|;|:| )")


def _strip_integral(s: str) -> str:
    """去掉 \\int、积分上下限、dx、以及各种排版噪音。"""
    # 用 (?![A-Za-z]) 而不是 \b:\b 在 "t" 和 "_" 之间不成立(下划线是单词字符),
    # 于是 \int_{0}^{1} 里的 \int 会完好无损地留下来。
    s = re.sub(r"\\i{0,3}int(?![A-Za-z])", " ", s)
    s = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{\s*d\s*\}", " ", s)
    s = _NOISE.sub(" ", s)
    # 去掉积分变量的微分符号 dx / dt / d x / 2dx
    # 用 (?<![a-zA-Z]) 而不是 \b,否则 "2dx" 这种紧贴数字的写法漏掉
    s = re.sub(r"(?<![a-zA-Z])d\s*([a-zA-Z])\b", " ", s)
    return s


def _normalize(s: str, strip_integral: bool = True) -> str:
    """把 LaTeX 逐轮降级成 Python 表达式字符串,直到不再变化。

    strip_integral=False 用于解析积分上下限(限里不会有 \\int 或 dx,
    但会有 \\infty、\\pi、\\frac,那些还是要处理)。
    """
    s = _strip_integral(s) if strip_integral else s
    s = (s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
          .replace("\\pi", "pi").replace("\\infty", "oo"))

    for _ in range(40):
        before = s
        s = _expand_frac(s)
        s = _expand_sqrt(s)
        s = _expand_functions(s)
        s = _expand_abs(s)
        if s == before:
            break

    # 剩下的花括号只是分组:^{2} -> **(2),e^{x} -> E**(x)
    s = s.replace("^{", "**(").replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    # 单独出现的 e 视作自然常数(前面已用词边界避开 sec / exp 这类词)
    s = re.sub(r"(?<![A-Za-z_0-9])e(?![A-Za-z_0-9])", "E", s)
    s = s.strip()

    if "\\" in s:
        raise ParseError(f"规范化后仍残留无法识别的 LaTeX 命令:{s!r}")
    return s


# ---------------------------------------------------------------- 主入口
_INT_HEAD = re.compile(r"\\int(?![A-Za-z])\s*(?:\\limits\s*)?")


def _extract_limits(s: str) -> tuple[str | None, str | None, str]:
    """从 `\\int_{a}^{b}` / `\\int_a^b` 里取出上下限,并把这整段从串里摘掉。

    必须在 `_strip_integral` 之前做 —— 后者只是把 `\\int` 抹掉,
    留下的 `_{0}^{1}` 会被后续的 `{`→`(` 替换变成 `_(0)**(1)`,
    然后解析就炸了。

    支持:上下限任意顺序、带不带花括号、`\\limits`、以及不定积分(没有_limit)。
    """
    match = _INT_HEAD.search(s)
    if not match:
        return None, None, s

    i = match.end()
    lower = upper = None

    for _ in range(2):
        while i < len(s) and s[i] == " ":
            i += 1
        if i >= len(s) or s[i] not in "_^":
            break
        kind = s[i]
        i += 1
        while i < len(s) and s[i] == " ":
            i += 1
        if i < len(s) and s[i] == "{":
            token, i = _read_balanced(s, i, "{", "}")
        else:
            start = i
            while i < len(s) and s[i] not in "_^{} ":
                i += 1
            token = s[start:i]
        if not token:
            break
        if kind == "_":
            lower = token
        else:
            upper = token

    rest = s[:match.start()] + " " + s[i:]
    return lower, upper, rest


def parse_limit(text: str) -> sp.Expr:
    """解析一个积分限,如 0、\\pi、-\\infty、\\frac{\\pi}{2}。"""
    if text is None:
        raise ParseError("积分限为空")
    raw = _normalize(text, strip_integral=False)
    if not raw:
        raise ParseError(f"积分限清洗后为空:{text!r}")
    _guard(raw)
    local = dict(SYMBOLS)
    local.update({"E": sp.E, "pi": sp.pi, "oo": sp.oo, "sqrt": sp.sqrt, "Abs": sp.Abs})
    try:
        return _assume_real(parse_expr(
            raw, local_dict=local,
            transformations=_TRANSFORMS
            + (implicit_multiplication_application,), evaluate=True))
    except Exception as exc:
        raise ParseError(f"无法解析积分限 {text!r}(规范化后 {raw!r}):{exc}") from exc


@dataclass
class IntegralInput:
    """解析结果:被积函数 + 积分变量 + 可选的上下限。"""

    integrand: sp.Expr
    variable: sp.Symbol
    lower: sp.Expr | None = None
    upper: sp.Expr | None = None

    @property
    def is_definite(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def limits(self) -> tuple:
        return (self.variable, self.lower, self.upper)

    def describe(self) -> str:
        if self.is_definite:
            return f"∫_{{{self.lower}}}^{{{self.upper}}} {self.integrand} d{self.variable}"
        return f"∫ {self.integrand} d{self.variable}"


def _parse_expr_body(body: str) -> sp.Expr:
    raw = _normalize(body)
    if not raw:
        raise ParseError(f"清洗后为空,原始输入:{body!r}")
    _guard(raw)
    local = dict(SYMBOLS)
    local.update({"E": sp.E, "pi": sp.pi, "oo": sp.oo, "sqrt": sp.sqrt, "Abs": sp.Abs})
    transformations = _TRANSFORMS + (implicit_multiplication_application,)
    try:
        expr = _assume_real(parse_expr(raw, local_dict=local,
                                       transformations=transformations, evaluate=True))
    except Exception as exc:
        raise ParseError(f"无法解析 {body!r}\n  规范化后: {raw}\n  原因: {exc}") from exc
    if not isinstance(expr, sp.Expr):
        raise ParseError(f"解析结果不是表达式:{expr!r}")
    return expr


def parse_integral(text: str, var: str | None = None) -> IntegralInput:
    """解析不定积分或定积分。

    >>> parse_integral(r"\\int_{0}^{\\pi} \\sin x\\,dx").is_definite
    True
    """
    if not text or not text.strip():
        raise ParseError("输入为空")

    lower_raw, upper_raw, body = _extract_limits(text)
    expr = _parse_expr_body(body)

    if var is not None:
        symbol = SYMBOLS.get(var) or sp.Symbol(var, real=True)
        if symbol not in expr.free_symbols and expr.free_symbols:
            raise ParseError(f"表达式中找不到积分变量 {var}")
    else:
        symbol = _pick_variable(expr)

    lower = parse_limit(lower_raw) if lower_raw is not None else None
    upper = parse_limit(upper_raw) if upper_raw is not None else None
    if (lower is None) != (upper is None):
        raise ParseError(f"积分限不完整:下限 {lower_raw!r},上限 {upper_raw!r}")

    return IntegralInput(expr, symbol, lower, upper)


def parse_integrand(text: str, var: str | None = None) -> tuple[sp.Expr, sp.Symbol]:
    """把用户输入解析成 (被积函数, 积分变量)。定积分请用 parse_integral。

    >>> parse_integrand(r"\\int \\frac{1}{1+x^2} dx")
    (1/(x**2 + 1), x)
    """
    parsed = parse_integral(text, var)
    if parsed.is_definite:
        raise ParseError(
            f"这是定积分(限 {parsed.lower} → {parsed.upper}),请用 parse_integral()")
    return parsed.integrand, parsed.variable


def _pick_variable(expr: sp.Expr) -> sp.Symbol:
    """猜测积分变量:优先 x,否则取唯一自由符号。"""
    free = expr.free_symbols
    if DEFAULT_VAR in free:
        return DEFAULT_VAR
    reals = sorted((s for s in free if s.is_real), key=lambda s: s.name)
    if len(reals) == 1:
        return reals[0]
    if not free:
        # 被积函数是常数:默认对 x 积分
        return DEFAULT_VAR
    if len(free) == 1:
        return next(iter(free))
    raise ParseError(
        f"无法确定积分变量,表达式含多个符号:{sorted(s.name for s in free)};"
        f"请用 var= 显式指定")


def parse_or_none(text: str, var: str | None = None):
    """宽松版本:解析失败返回 None,用于批量校验语料。"""
    try:
        return parse_integrand(text, var)
    except ParseError:
        return None


def parse_any_or_none(text: str, var: str | None = None) -> IntegralInput | None:
    """宽松版本:不定积分和定积分都接受,返回 IntegralInput。"""
    try:
        return parse_integral(text, var)
    except ParseError:
        return None
