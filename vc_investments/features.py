"""Ranking-panel helpers from ``TCG_Data_preparation.ipynb``.

These are used when the original pickle extracts are available. They are kept
vectorized where that does not change the original definitions.
"""

from __future__ import annotations

import nltk
import numpy as np
import pandas as pd

SUBSCRIPTION_KEYWORDS = ("subscription", "renewal", "subscribe")


def ensure_nltk_punkt() -> None:
    """Download tokenizer data used by the original ``sub_finder``."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def sub_finder(frame: pd.DataFrame, column: str = "description") -> list[int]:
    """Flag apps whose description mentions subscription language.

    Same keyword list as the 2019 notebook. Non-string / missing descriptions
    are treated as empty so a single NaN does not abort the run.
    """
    ensure_nltk_punkt()
    flags: list[int] = []
    keywords = set(SUBSCRIPTION_KEYWORDS)
    for value in frame[column]:
        text = value if isinstance(value, str) else ""
        tokens = nltk.word_tokenize(text)
        flags.append(int(any(word.lower() in keywords for word in tokens)))
    return flags


def rank_growth(rank_panel: pd.DataFrame) -> list[float]:
    """Worst rank minus rank on the last observed day (notebook definition)."""
    growth: list[float] = []
    for _, row in rank_panel.iterrows():
        growth.append(float(np.max(row) - row.iloc[-3]))
    return growth


def top_rank_days(rank_panel: pd.DataFrame) -> list[int]:
    """Count of days spent at each app's best (minimum) rank."""
    days: list[int] = []
    for _, row in rank_panel.iterrows():
        best = row.min(skipna=True)
        days.append(int(row.value_counts()[best]))
    return days


def impute_rank_panel(rank_panel: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill along each app's rank history."""
    filled = rank_panel.ffill(axis=1)
    return filled.bfill(axis=1)
