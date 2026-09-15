"""兼容层:统计与评估的实现已经搬到独立的 llm-eval-toolkit 仓库。

`evallib.stats` 这个名字保留下来,是为了让本仓库里原有的
`from evallib import stats` 不用改 —— 但**实现只有一份**,在那个包里。

解析顺序(找到第一个可用为止):

  1. 已安装的 `llm_eval_toolkit`(推荐:`pip install -e ../llm-eval-toolkit`)
  2. 环境变量 `LLM_EVAL_TOOLKIT_SRC` 指向的 `src` 目录
  3. 同级的 `../llm-eval-toolkit/src`(开发期两个仓库并排放时的默认)

三条都不通就抛 ImportError 并说清怎么装 —— **不静默降级**。
一个悄悄变成空壳的依赖,比一个明确的报错难查得多。

为什么要独立成库:统计与评估协议跟"符号积分"这件事毫无关系,
它有自己的一批使用者、自己的一套测试、自己该用的名字。
放在这里只会让它看起来像这个项目的一部分。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _candidate_src_dirs() -> list[Path]:
    dirs: list[Path] = []
    from_env = os.environ.get("LLM_EVAL_TOOLKIT_SRC")
    if from_env:
        dirs.append(Path(from_env))
    dirs.append(_REPO.parent / "llm-eval-toolkit" / "src")
    return dirs


def _ensure_importable() -> None:
    try:
        import llm_eval_toolkit  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in _candidate_src_dirs():
        if (candidate / "llm_eval_toolkit" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise ImportError(
        "找不到 llm-eval-toolkit。任选一种:\n"
        "  1. pip install -e ../llm-eval-toolkit   (开发期推荐)\n"
        "  2. pip install llm-eval-toolkit\n"
        "  3. 设环境变量 LLM_EVAL_TOOLKIT_SRC=<...>/llm-eval-toolkit/src\n"
        "统计与评估的实现已从本仓库搬走,详见 eval/FINDINGS.md。"
    )


_ensure_importable()

from llm_eval_toolkit import (  # noqa: E402,F401
    build_evalset, metrics, report, stats,
)

__all__ = ["stats", "metrics", "build_evalset", "report"]
