"""Train the carried-forward model and report validation metrics.

Run from the repo root:  uv run python -m bankguard.train
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.lightgbm

from bankguard.data import load_accounts, load_transactions, split_by_date
from bankguard.evaluate import evaluate
from bankguard.features import (
    CATEGORICAL,
    FEATURES,
    build_features,
    fit_currency_median,
)
from bankguard.model import PARAMS, fit_categories, fit_gbm, predict


def main(data_dir: Path, seed: int = 42) -> dict:
    tx = load_transactions(data_dir / "HI-Small_Trans.csv")
    accounts = load_accounts(data_dir / "HI-Small_accounts.csv")
    splits = split_by_date(tx)
    train, val = splits["train"], splits["validation"]

    # Learned from train only, then applied to every row.
    currency_median = fit_currency_median(train)
    X = build_features(tx, accounts, currency_median)

    X_train, y_train = X.loc[train.index], train["is_laundering"].to_numpy()
    X_val, y_val = X.loc[val.index], val["is_laundering"].to_numpy()
    categories = fit_categories(X_train, CATEGORICAL)

    model = fit_gbm(X_train, y_train, X_val, y_val, FEATURES, categories, seed=seed)
    scores = predict(model, X_val, FEATURES, categories)

    result = evaluate(y_val, scores, "LightGBM, owner attributes")
    result["trees"] = model.best_iteration_

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bankguard-laundering")
    with mlflow.start_run(run_name=result["model"]):
        mlflow.log_params({**PARAMS, "seed": seed, "n_features": len(FEATURES)})
        mlflow.log_metrics({k.replace("%", "pct"): v for k, v in result.items() if k != "model"})
        # Everything needed to score a new transaction the same way as in training.
        mlflow.log_dict(
            {"features": FEATURES, "categories": categories, "currency_median": currency_median},
            "preprocessing.json",
        )
        mlflow.lightgbm.log_model(model, name="model")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for key, value in main(args.data_dir, args.seed).items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")