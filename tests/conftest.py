"""测试夹具。

只提供两样东西:

1. `workdir` —— 一个普通的工作目录,替代 pytest 的 `tmp_path`。
   这里的用例需要往磁盘上写评估报告 JSON,而 `tmp_path` 依赖 pytest 的
   tmpdir 插件;在受限环境里那套"编号目录 + 锁文件"机制会把 basetemp 变成
   **读不回来也删不掉**的状态(实测复现两次,连 takeown / icacls 都被拒)。
   `workdir` 只用最朴素的 mkdir,并且每次用独立的随机目录名,
   某一次留下的坏目录不会殃及后续运行。

2. `anyio_backend` —— 占位,避免第三方异步插件在收集阶段报错。

受限环境下的运行方式(本机沙箱实测):

    pytest tests/ -p no:cacheprovider -p no:tmpdir

`-p no:cacheprovider` 是因为 pytest 的缓存写入会被拒;
`-p no:tmpdir` 是因为 basetemp 会被污染。**这两条只在本机沙箱里需要**,
正常终端直接 `pytest tests/` 即可 —— 所以不把它们写进 pytest.ini,
免得在一个正常环境里也被强加上去。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def workdir():
    """一次性的工作目录,用例结束后自动清理。"""
    d = REPO / f".pytest-work-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=False)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def anyio_backend():                # pragma: no cover
    return "asyncio"
