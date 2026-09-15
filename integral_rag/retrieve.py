"""检索层 —— 从方法卡片库里找出"这道题该用什么方法"。

三个打分通道,最后加权求和:

  A. 触发条件匹配(主通道)
     卡片自己声明了适用条件(triggers),比如「含 √(a²−x²)」。
     这是可解释的:命中/未命中哪一条,都能打印出来给人看。
     其中 has_radical 是硬约束 —— 被积函数里没有对应的根号,
     那么三角替换卡片就是不相干的,直接排除,不给它靠别的通道翻盘的机会。

  B. 例题结构相似度(辅通道)
     把卡片自带的例题和被积函数都表示成"节点类型直方图",算余弦相似度。
     结构相似度对积分是有意义的:节点组成相近的式子,方法往往也相近。
     注意这是**结构**相似,不是字符相似 —— 这正是它比文本检索好用的地方。

  C. 中文文本检索(可选通道)
     用汉字二元组 + BM25。用于"根号里面是平方差怎么做"这类
     只给了语言描述、没给式子的问法。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp

from .features import Features, extract, extract_definite, structural_similarity
from .parse import parse_any_or_none, parse_or_none

# A 通道里各字段的权重
_WEIGHTS = {"interval": 2.5, "has_radical": 3.0, "misc": 2.0,
            "has_func": 1.2, "form": 1.0}
# 命中不了就硬性排除的字段
_HARD_FIELDS = {"has_radical"}
# 软性未命中的惩罚系数,按字段分开给。
#
# misc 罚得重:它是最"具体"的字段,"多项式×指数"这种组合信号一旦对不上,
#   基本就说明方法不适用。原来一律只罚 0.2,结果 ∫x·e^{x²}dx 里
#   parts_tabular 的 poly_times_exp 明明没命中,却靠 has_func 的覆盖率
#   把凑微分卡片压了下去 —— 而这道题正确的是凑微分,分部积分做不动。
#
# interval 罚得也重:区间是定积分方法的第一决定因素。
#   卡片写的是"对称区间上的奇偶性",题目区间不对称就该排到后面去。
#
# has_func / form 罚得轻:它们是宽泛的分类。查询里没有某个信号,对这些
#   宽卡片的否定力度本来就弱(∫ln x dx 没有多项式,但分部积分恰恰是对的)。
_SOFT_MISS = {"interval": 0.5, "misc": 0.55, "has_func": 0.2, "form": 0.2}
_DEFAULT_SOFT_MISS = 0.2
# 卡片自带的例题和当前被积函数结构几乎一致时给的加成。
# 这是整个检索里最强的证据:出卡片的人把这道题当范例写进去了。
#
# 阈值必须卡得很紧。之前把"接近"的门槛放在 0.9,结果 ∫dx/(2x+5) 和例题
# 1/(x²−1) 的节点直方图余弦是 0.962 —— 两者的节点组成确实像,但方法毫无关系
# (一个凑微分、一个部分分式),白白送出 1.5 分把部分分式推上了第一。
# 现在只有结构上真正相等的例题才算"命中"。
_EXACT_EXAMPLE_BONUS = 3.0
_NEAR_EXAMPLE_BONUS = 1.0
_EXACT_COSINE = 0.999
_NEAR_COSINE = 0.995

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "method_cards.json"


@dataclass
class Hit:
    card: dict
    score: float
    matched: list[str] = field(default_factory=list)
    example_sim: float = 0.0
    max_example_cosine: float = 0.0
    text_score: float = 0.0
    # 分数拆解,便于回答"为什么推荐了这张卡"
    parts: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.card.get("id", "?")

    @property
    def name(self) -> str:
        return self.card.get("name", self.id)

    @property
    def family(self) -> str:
        return self.card.get("family", "")

    def explain(self) -> str:
        bits = [f"{key} {value:+.2f}" for key, value in self.parts.items()]
        return f"{self.id} 总分 {self.score:+.2f} = " + " ".join(bits)


# ---------------------------------------------------------------- 语料加载
class Retriever:
    def __init__(self, cards: list[dict], definite: bool = False):
        self.cards = cards
        self.definite = definite
        self._example_cache: dict[str, list[tuple[Counter, Features]]] = {}
        self._bm25 = _BM25([_card_text(c) for c in cards]) if cards else None
        for card in cards:
            self._example_cache[card.get("id", "")] = self._parse_examples(card)

    # ---------------------------------------------------------- 加载
    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CORPUS,
                  definite: bool = False) -> "Retriever":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到方法卡片语料:{path}\n"
                f"请先运行 python build_corpus.py 生成/校验语料。")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} 的顶层应该是 JSON 数组")
        return cls(data, definite=definite)

    def _parse_examples(self, card: dict) -> list[tuple[Counter, Features]]:
        """把卡片里的例题 LaTeX 解析成结构指纹,解析失败的例题直接跳过。

        定积分卡片的例题是**完整定积分**(带上下限),所以要走 parse_integral;
        不定积分卡片走 parse_integrand。两条路的特征提取也不同
        (定积分要额外提取区间类型)。
        """
        out: list[tuple[Counter, Features]] = []
        for raw in card.get("examples") or []:
            try:
                if self.definite:
                    parsed = parse_any_or_none(raw)
                    if parsed is None or not parsed.is_definite:
                        continue
                    feats = extract_definite(parsed.integrand, parsed.variable,
                                             parsed.lower, parsed.upper)
                else:
                    parsed = parse_or_none(raw)
                    if parsed is None:
                        continue
                    expr, var = parsed
                    feats = extract(expr, var)
            except Exception:
                continue
            out.append((feats.tokens, feats))
        return out

    # ---------------------------------------------------------- 主入口
    def retrieve(self, feats: Features, k: int = 4,
                 query_text: str | None = None,
                 vector_weight: float = 1.0,
                 text_only: bool = False) -> list[Hit]:
        """检索方法卡片。

        text_only=True 时只走中文文本通道,用于"根号里是平方差怎么做"
        这类只给语言描述、没给式子的问法 —— 此时没有任何结构特征,
        触发条件通道(尤其是 has_radical 的硬约束)必须整个跳过,
        否则所有带根号的卡片都会被硬性排除。
        """
        text_scores = self._bm25.scores(query_text) if (query_text and self._bm25) else None

        hits: list[Hit] = []
        for index, card in enumerate(self.cards):
            if text_only:
                score, matched, rejected = 0.0, [], False
                example_sim = max_cosine = example_bonus = 0.0
            else:
                triggers = card.get("triggers") or {}
                score, matched, rejected = _match_triggers(feats, triggers)
                if rejected:
                    continue
                example_sim, max_cosine = self._example_similarity(card, feats)
                if max_cosine >= _EXACT_COSINE:
                    example_bonus = _EXACT_EXAMPLE_BONUS
                    matched.append(f"★例题结构完全吻合(cos={max_cosine:.3f})")
                elif max_cosine >= _NEAR_COSINE:
                    example_bonus = _NEAR_EXAMPLE_BONUS
                    matched.append(f"✓例题结构高度接近(cos={max_cosine:.3f})")
                else:
                    example_bonus = 0.0

            text_score = 0.0
            if text_scores is not None:
                # BM25 分数范围不定,压缩到 0~3 再当加权项
                text_score = 3.0 * text_scores[index] / (1.0 + text_scores[index])

            try:
                priority = float(card.get("priority", 50))
            except Exception:
                priority = 50.0

            example_term = 2.0 * example_sim * vector_weight
            priority_term = 0.0 if text_only else 0.01 * priority
            total = score + example_term + example_bonus + text_score - priority_term
            hits.append(Hit(card=card, score=total, matched=matched,
                            example_sim=example_sim, max_example_cosine=max_cosine,
                            text_score=text_score,
                            parts={"触发条件": score,
                                   "例题相似": example_term,
                                   "例题命中": example_bonus,
                                   "文本": text_score,
                                   "优先级": -priority_term}))

        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    @staticmethod
    def _same_shape(query: Features, example: Features) -> bool:
        """两道题的结构特征是否一致 —— "例题命中"加分的**门禁**。

        为什么要门禁:小表达式的节点直方图会饱和。`x³e^{-x}` 和 `e^{-x²}`
        的直方图余弦是 **1.000**,于是"结构完全吻合"的 +3.0 分被误发给
        一张**方法完全不同**的卡片(实测把 Γ 函数形状判成了高斯)。

        签名取 has_func / form / interval 三项:
          * has_func、form 区分"是不是同一类被积函数";
          * interval 让**区间真的参与最强的那条通道** ——
            否则 `∫_{-a}^{a} x²dx` 与 `∫_0^1 x²dx` 会拿到同一个满分,
            而后者区间根本不对称(实测 `∫_0^1 x²dx` 被判给了"对称性")。
        """
        a = query.as_match_dict()
        b = example.as_match_dict()
        for key in ("has_func", "form", "interval"):
            if set(a.get(key) or ()) != set(b.get(key) or ()):
                return False
        return True

    def _example_similarity(self, card: dict, feats: Features) -> tuple[float, float]:
        """返回 (综合例题相似度, 最高的节点直方图余弦)。

        相似度 = 0.5·节点直方图余弦 + 0.5·触发条件重合度。
        只用直方图会被"多项式次数多一项"这类噪声干扰,混入触发条件重合度
        之后,判断更贴近"方法是否相同"。
        第二个返回值单独给出来,是为了让"例题几乎就是这道题"这种强证据
        能被识别出来并单独加分 —— 它**必须**先通过 `_same_shape` 门禁:
        "吻合"不能只看一个会饱和的余弦。
        """
        examples = self._example_cache.get(card.get("id", ""), [])
        if not examples:
            return 0.0, 0.0
        best = 0.0
        best_cosine = 0.0
        for tokens, ex_feats in examples:
            cosine = structural_similarity(feats.tokens, tokens)
            overlap, _, _ = _match_triggers(feats, ex_feats.as_match_dict())
            overlap_norm = max(0.0, overlap) / 6.0      # 6.0 是满配时的量级
            best = max(best, 0.5 * cosine + 0.5 * overlap_norm)
            if self._same_shape(feats, ex_feats):
                best_cosine = max(best_cosine, cosine)
        return min(best, 1.5), best_cosine


# ---------------------------------------------------------------- 触发条件匹配
def _match_triggers(feats: Features, triggers: dict) -> tuple[float, list[str], bool]:
    """返回 (得分, 人类可读的命中说明, 是否硬性排除)。

    打分同时看两个方向:
      coverage  = 卡片声明的条件里,有多少被满足(卡片视角)
      precision = 当前被积函数的特征里,有多少被这张卡片解释(题目视角)
    只算 coverage 会让"声明了一大堆条件的宽卡片"占便宜;
    只算 precision 会让"随便一张卡片"都能拿分。两个一起看才稳。
    """
    query = feats.as_match_dict()
    total = 0.0
    matched: list[str] = []

    for key, weight in _WEIGHTS.items():
        want = set(triggers.get(key) or [])
        if not want or "any" in want:
            continue
        got = set(query.get(key) or set())

        # has_radical 里的 "none" 表示"不含根式"
        if key == "has_radical" and want == {"none"}:
            if not got:
                total += weight
                matched.append("✓无根式")
            else:
                return 0.0, [f"✗要求无根式,但检出 {sorted(got)}"], True
            continue

        inter = want & got
        if not inter:
            if key in _HARD_FIELDS:
                return 0.0, [f"✗缺少必需条件 {key}={sorted(want)}"], True
            total -= weight * _SOFT_MISS.get(key, _DEFAULT_SOFT_MISS)
            matched.append(f"✗{key}未命中(要求 {sorted(want)})")
            continue

        coverage = len(inter) / len(want)
        precision = len(inter) / len(got) if got else 0.0
        total += weight * (coverage + 0.5 * precision)
        matched.append(f"✓{key}:{'/'.join(sorted(inter))}"
                       f"(覆盖{coverage:.0%}/解释{precision:.0%})")

    return total, matched, False


# ---------------------------------------------------------------- BM25(中文二元组)
def _tokenize(text: str) -> list[str]:
    clean = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    if len(clean) < 2:
        return [clean] if clean else []
    return [clean[i:i + 2] for i in range(len(clean) - 1)]


class _BM25:
    """极简 BM25。中文用汉字二元组切分,和 rag_v2.py 里的离线思路一致。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.avg_len = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for counter in self.tf:
            for term in counter:
                df[term] += 1
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = _tokenize(query)
        out = [0.0] * self.n
        for i, counter in enumerate(self.tf):
            length = len(self.docs[i]) or 1
            score = 0.0
            for term in terms:
                if term not in counter:
                    continue
                freq = counter[term]
                score += (self.idf.get(term, 0.0) * freq * (self.k1 + 1)
                          / (freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)))
            out[i] = score
        return out


def _card_text(card: dict) -> str:
    fields = [card.get("name", ""), card.get("when", ""), card.get("recipe", ""),
              card.get("pitfalls", ""), " ".join(card.get("examples") or [])]
    return " ".join(str(f) for f in fields)


# ---------------------------------------------------------------- 自检
def describe_hits(hits: list[Hit], limit: int = 4) -> str:
    lines = []
    for rank, hit in enumerate(hits[:limit], start=1):
        lines.append(f"  {rank}. [{hit.score:6.2f}] {hit.id} — {hit.name}")
        if hit.matched:
            lines.append(f"       触发:{' '.join(hit.matched)}")
        lines.append(f"       例题结构相似度 {hit.example_sim:.3f}")
    return "\n".join(lines)
