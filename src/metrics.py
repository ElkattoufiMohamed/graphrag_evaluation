from __future__ import annotations
from typing import List, Tuple
import re
from rouge_score import rouge_scorer

def _tok(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []

def token_f1(pred: str, gold: str) -> float:
    p = _tok(pred)
    g = _tok(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_set = {}
    for t in p: p_set[t] = p_set.get(t, 0) + 1
    g_set = {}
    for t in g: g_set[t] = g_set.get(t, 0) + 1

    overlap = 0
    for t, c in p_set.items():
        overlap += min(c, g_set.get(t, 0))

    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

class RougeL:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, pred: str, gold: str) -> float:
        return float(self.scorer.score(gold, pred)["rougeL"].fmeasure)

def best_of_many(pred: str, gold_list: List[str]) -> Tuple[float, float]:
    """Return (best_f1, best_rougeL) across multiple gold answers."""
    rl = RougeL()
    best_f1 = 0.0
    best_rl = 0.0
    for g in gold_list:
        best_f1 = max(best_f1, token_f1(pred, g))
        best_rl = max(best_rl, rl.score(pred, g))
    return best_f1, best_rl
