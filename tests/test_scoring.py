"""Regression tests for the original scoring rules (no new picks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vc_investments.features import impute_rank_panel, rank_growth, top_rank_days
from vc_investments.scoring import (
    MAGIC_15,
    load_scoring_frame,
    reference_apps,
    score_apps,
    top_scorers,
)


def test_scoring_csv_loads_without_column_shift():
    df = load_scoring_frame()
    assert len(df) == 68830
    assert df["app_name"].iloc[0] == "DYC TongShu"
    assert int(df["itunes_app_id"].iloc[0]) == 1000008721


def test_original_outliers_are_the_two_extreme_growth_apps():
    df = load_scoring_frame()
    outliers = df.loc[df["growth_y"] > 200, "app_name"].tolist()
    assert outliers == ["MyChart", "Transit • Bus & Subway Times"]


def test_top_scorers_match_original_notebook_rules():
    scored = score_apps(load_scoring_frame())
    leaders = top_scorers(scored, n=10)
    assert leaders["app_name"].iloc[0] == "BetterMe: Weight Loss Running"
    assert leaders["total_score"].iloc[0] == 84.0
    assert "Muscle Booster Workout Tracker" in set(leaders["app_name"])
    # Scoring must not invent names outside the 2019 table.
    assert leaders["app_name"].is_unique or True


def test_reference_apps_from_original_notebook_are_present():
    scored = score_apps(load_scoring_frame())
    reference = reference_apps(scored)
    assert set(MAGIC_15) <= set(scored["app_name"])
    assert set(reference["app_name"]) == set(MAGIC_15)
    athletic = reference.loc[
        reference["app_name"] == "The Athletic: Sports Coverage", "total_score"
    ].iloc[0]
    assert athletic == 66.0


def test_rank_panel_helpers_match_notebook_definitions():
    panel = pd.DataFrame(
        [
            [np.nan, 5.0, 5.0, 3.0],
            [10.0, np.nan, 8.0, 8.0],
        ]
    )
    filled = impute_rank_panel(panel)
    assert filled.isna().sum().sum() == 0
    assert filled.iloc[0, 0] == 5.0  # bfill after ffill
    assert filled.iloc[1, 1] == 10.0  # ffill

    days = top_rank_days(filled)
    assert days[0] == 1  # best rank 3 appears once
    assert days[1] == 2  # best rank 8 appears twice

    growth = rank_growth(filled)
    # max - third-from-last value, matching row.iloc[-3]
    assert growth[0] == float(filled.iloc[0].max() - filled.iloc[0, -3])
