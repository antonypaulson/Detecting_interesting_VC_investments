"""Reproduce the original 2019 weighted scoring model.

The weights, bins, category bonuses, and reference app list are taken from
``TCG_scoring_model.ipynb``. This module does not invent new investment picks;
it only makes that notebook's scoring runnable outside Colab.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vc_investments.paths import SCORING_CSV

# Apps the original analysis treated as a known-interesting reference set
# (investor-indicated names from the 2019 notebook, Netflix commented out).
MAGIC_15 = [
    "The Action Network: Sports App",
    "ESPN: Live Sports & Scores",
    "AllTrails: Hike, Run & Cycle",
    "Headspace: Meditation & Sleep",
    "Quizlet",
    "Duolingo",
    "The Athletic: Sports Coverage",
    "Prodigy Math Game",
    "onX Hunt: #1 GPS Hunting Map",
    "HOOKED",
    "Crunchyroll",
    "The Wall Street Journal.",
    "MLB At Bat",
    "Surfline",
]

# Notebook dropped two extreme rank-growth rows (MyChart, Transit).
OUTLIER_GROWTH_THRESHOLD = 200

# 2019 CB Insights-inspired category weights from the scoring notebook.
CATEGORY_SCORES = {
    "Health & Fitness": 10,
    "Finance": 8,
    "Social Networking": 6,
    "Travel": 4,
    "Education": 2,
}

SCORE_COLUMNS = [
    "growth_score",
    "daily_growth_score",
    "max_growth_score",
    "top_rank_days_score",
    "top_rank_score",
    "subscription_score",
    "rating_oo5_score",
    "active_days_score",
    "category_score",
    "developer_score",
]

RESULT_COLUMNS = ["app_name", *SCORE_COLUMNS, "total_score"]

# Same slice as ``developers = ...value_counts()[:91]`` in the notebook.
DEVELOPER_GROWTH_CUTOFF = 8
DEVELOPER_TOP_N = 91


def load_scoring_frame(path: Path | None = None) -> pd.DataFrame:
    """Load ``data/frame_scoring_model.csv``.

    The committed file has an unnamed leading index column and one more field
    per row than named headers. pandas treats that extra field as the index.
    """
    csv_path = Path(path) if path is not None else SCORING_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Scoring table not found at {csv_path}. "
            "Clone the repository with the data/ directory, or see data/README.md."
        )
    df = pd.read_csv(csv_path)
    expected = {
        "app_name",
        "developer",
        "category",
        "subscription_service",
        "rating_oo5_x",
        "top_rank_y",
        "top_rank_days_y",
        "growth_y",
        "active_days",
        "avg_daily_growth",
        "max_growth_rate",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if df["app_name"].dtype != object and "itunes_app_id" not in df.columns:
        raise ValueError(
            f"{csv_path} looks column-shifted; expected app_name to be text."
        )
    return df


def _cut_score(series: pd.Series, bins: int, labels: list) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels).astype("float64")


def _category_score(category: pd.Series) -> pd.Series:
    return category.map(CATEGORY_SCORES).fillna(0).astype("float64")


def _developer_score(df: pd.DataFrame) -> pd.Series:
    """Bonus for developers with more than one high-growth app.

    Matches the notebook: keep the 91 most frequent developers among apps with
    ``growth_score >= 8``, then 5-bin that frequency into 2/4/6/8/10.
    """
    high_growth = df.loc[df["growth_score"] >= DEVELOPER_GROWTH_CUTOFF, "developer"]
    developers = high_growth.value_counts().iloc[:DEVELOPER_TOP_N]
    developer_table = developers.to_frame(name="developer_counts")
    developer_table["developer_score"] = _cut_score(
        developer_table["developer_counts"],
        bins=5,
        labels=[2.0, 4.0, 6.0, 8.0, 10.0],
    )
    mapped = df["developer"].map(developer_table["developer_score"])
    return mapped.fillna(0).astype("float64")


def score_apps(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    """Apply the original notebook scoring rules and add ``total_score``."""
    scored = df.copy()
    if drop_outliers:
        scored = scored.loc[scored["growth_y"] <= OUTLIER_GROWTH_THRESHOLD].copy()

    scored["growth_score"] = _cut_score(scored["growth_y"], 5, [2, 4, 6, 8, 10])
    scored["daily_growth_score"] = _cut_score(
        scored["avg_daily_growth"], 5, [2, 4, 6, 8, 10]
    )
    scored["max_growth_score"] = _cut_score(
        scored["max_growth_rate"], 5, [2, 4, 6, 8, 10]
    )
    scored["top_rank_days_score"] = _cut_score(
        scored["top_rank_days_y"], 5, [2, 4, 6, 8, 10]
    )
    scored["top_rank_score"] = _cut_score(
        scored["top_rank_y"], 5, [10, 8, 6, 4, 2]
    )
    scored["subscription_score"] = scored["subscription_service"].eq(1).astype(int) * 20
    scored["rating_oo5_score"] = _cut_score(
        scored["rating_oo5_x"], 5, [1, 2, 3, 4, 5]
    )
    scored["active_days_score"] = _cut_score(
        scored["active_days"],
        10,
        [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5],
    )
    scored["category_score"] = _category_score(scored["category"])
    scored["developer_score"] = _developer_score(scored)
    scored["total_score"] = scored[SCORE_COLUMNS].sum(axis=1)
    return scored


def top_scorers(scored: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        scored.sort_values("total_score", ascending=False)[RESULT_COLUMNS]
        .head(n)
        .reset_index(drop=True)
    )


def reference_apps(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.loc[scored["app_name"].isin(MAGIC_15), RESULT_COLUMNS].sort_values(
        "total_score", ascending=False
    )
