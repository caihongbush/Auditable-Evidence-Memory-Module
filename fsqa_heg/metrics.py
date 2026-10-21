"""轻量 QA 指标（不依赖 bert_score / torch）。"""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_char(prediction: str, ground_truth: str) -> float:
    """中文等无空格文本：按字符集合算 F1。"""
    pred = normalize_answer(prediction).replace(" ", "")
    gold = normalize_answer(ground_truth).replace(" ", "")
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    p = common / len(pred)
    r = common / len(gold)
    return (2 * p * r) / (p + r)


def f1(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    token_scores = [
        max(f1_score(prediction, gt) for prediction in predictions)
        for gt in ground_truths
    ]
    char_scores = [
        max(f1_char(prediction, gt) for prediction in predictions)
        for gt in ground_truths
    ]
    return float(np.mean([max(a, b) for a, b in zip(token_scores, char_scores)]))
