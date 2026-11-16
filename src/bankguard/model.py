"""LightGBM training for the laundering model."""

from typing import Any

import lightgbm as lgb
import pandas as pd


def fit_categories(X_train: pd.DataFrame, columns: list[str]) -> dict[str, list]:
    """The allowed values of each categorical column, learned from train only."""
    return {c: sorted(X_train[c].dropna().unique()) for c in columns}


def gbm_frame(X: pd.DataFrame, columns: list[str], categories: dict[str, list]) -> pd.DataFrame:
    """Select the model's columns, in a fixed order, with fixed category codes."""
    out = X[columns].copy()
    for c in columns:
        if c in categories:
            known = out[c].where(out[c].isin(categories[c]))
            out[c] = pd.Categorical(known, categories=categories[c])
    return out

PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "reg_lambda": 10.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "metric": "average_precision",
    "verbose": -1,
}


def fit_gbm(X_train, y_train, X_val, y_val, columns, categories, seed: int = 42):
    """Train LightGBM, stopping when validation average precision stops improving."""
    model = lgb.LGBMClassifier(**PARAMS, random_state=seed)
    model.fit(
        gbm_frame(X_train, columns, categories),
        y_train,
        eval_X=gbm_frame(X_val, columns, categories),
        eval_y=y_val,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict(model, X, columns, categories):
    """Laundering score between 0 and 1 for each row."""
    return model.predict_proba(gbm_frame(X, columns, categories))[:, 1]