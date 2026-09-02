"""Consistency model from ``EDA_and_Machine_Learning_TCG.ipynb``.

Trains the original regressors to predict ``top_rank_days`` from the processed
scoring table. ``is_editor_choice`` is omitted because that column was not
exported into the files shipped in this repository.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from vc_investments.scoring import OUTLIER_GROWTH_THRESHOLD, load_scoring_frame

# Original feature list minus ``is_editor_choice`` (not in the shipped CSV).
BASE_FEATURES = [
    "rating_oo5",
    "num_ratings",
    "has_iap",
    "file_size",
    "price",
    "subscription_service",
    "top_rank",
    "rank_growth",
    "n_languages",
    "active_days",
    "avg_daily_growth",
    "max_growth_rate",
]


def load_consistency_frame(path=None) -> pd.DataFrame:
    """Map scoring-CSV names onto the names used after EDA cleaning."""
    df = load_scoring_frame(path)
    df = df.loc[df["growth_y"] <= OUTLIER_GROWTH_THRESHOLD].copy()
    return df.rename(
        columns={
            "rating_oo5_x": "rating_oo5",
            "num_ratings_x": "num_ratings",
            "top_rank_y": "top_rank",
            "top_rank_days_y": "top_rank_days",
            "growth_y": "rank_growth",
        }
    )


def make_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dummy = pd.get_dummies(df[["category", "age_rating"]], drop_first=False)
    X = pd.concat([df[BASE_FEATURES], dummy], axis=1)
    y = df["top_rank_days"]
    return X, y, list(X.columns)


def train_baseline_models(random_state: int = 42):
    """Fit the original model family and return mean 5-fold CV R^2 on the test split.

    RandomForest uses ``n_estimators=10`` to match sklearn 0.20 (the Colab
    default when the notebook was written). Newer sklearn defaults to 100.
    """
    df = load_consistency_frame()
    X, y, _ = make_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "linear_regression": LinearRegression(),
        "knn": KNeighborsRegressor(),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=10, random_state=random_state
        ),
    }
    results = {}
    for name, model in models.items():
        if name in {"linear_regression", "knn"}:
            model.fit(X_train_sc, y_train)
            score = cross_val_score(model, X_test_sc, y_test, cv=5).mean()
        else:
            model.fit(X_train, y_train)
            score = cross_val_score(model, X_test, y_test, cv=5).mean()
        results[name] = float(score)
    return results
