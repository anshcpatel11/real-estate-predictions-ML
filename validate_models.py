"""
Data-driven checks: stratified CV, dummy baselines, and hold-out metrics.

Use this to sanity-check that learned models beat chance and that the hand-tuned
baseline is not worse than simpler weighting; CV reduces reliance on a single split.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, cross_validate

from baseline_scoring import (
    BASELINE_WEIGHTS,
    TOP_QUANTILE,
    baseline_cv_scores,
    baseline_score,
    baseline_used_columns,
    fit_zscore,
    signed_unit_weights,
    zscore_matrix,
)
from supervised_models import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    binary_metrics_dict,
    make_logistic_regression,
    make_random_forest,
    ml_feature_columns,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-validate models and compare to dummies.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--parallel-cv",
        action="store_true",
        help="Use n_jobs=-1 in cross_validate (can fail in some sandboxes). Default is single-threaded CV.",
    )
    args = p.parse_args()
    cv_n_jobs = -1 if args.parallel_cv else 1

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module="sklearn")

    train = pd.read_csv(args.train, low_memory=False)
    test  = pd.read_csv(args.test, low_memory=False)
    y_train = train["label"].to_numpy()
    y_test  = test["label"].to_numpy()

    def print_cv(label, scores):
        print(f"  {label}")
        for metric, vals in scores.items():
            print(f"    {metric:<12} {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── Dummy baselines ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  Dummy Baselines (5-fold CV)")
    print("─" * 60)
    for strat in ("most_frequent", "stratified"):
        dc = DummyClassifier(strategy=strat, random_state=args.random_state)
        cv_res = cross_validate(
            dc,
            np.zeros((len(train), 1)),
            y_train,
            cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state),
            scoring=("accuracy", "f1", "precision", "recall"),
            n_jobs=cv_n_jobs,
        )
        scores = {k.replace("test_", ""): v for k, v in cv_res.items() if k.startswith("test_")}
        print_cv(f"DummyClassifier ({strat})", scores)

    # ── Baseline (demographic index) ─────────────────────
    print("\n" + "─" * 60)
    print("  Demographic Index Baseline (5-fold CV)")
    print("─" * 60)
    used_cols = baseline_used_columns(train)
    unit_w    = signed_unit_weights(used_cols, BASELINE_WEIGHTS)
    for label, w in (
        ("Baseline (hand weights)", None),
        ("Baseline (±1 same signs)", unit_w),
    ):
        scores = baseline_cv_scores(
            train, y_train,
            n_splits=args.cv_folds,
            weights=w,
            random_state=args.random_state,
        )
        print_cv(label, scores)

    # ── ML models CV ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("  ML Models (5-fold CV on train set)")
    print("─" * 60)
    feat_cols = ml_feature_columns(train)
    X_train_arr = train[feat_cols].to_numpy(dtype=np.float64)
    X_test_arr  = test[feat_cols].to_numpy(dtype=np.float64)
    cv_split    = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    for label, est in (
        ("LogisticRegression", make_logistic_regression()),
        ("RandomForest",       make_random_forest(n_jobs=1)),
    ):
        cv_res = cross_validate(
            est, X_train_arr, y_train,
            cv=cv_split,
            scoring=("accuracy", "f1", "precision", "recall", "roc_auc"),
            n_jobs=cv_n_jobs,
        )
        scores = {k.replace("test_", ""): v for k, v in cv_res.items() if k.startswith("test_")}
        print_cv(label, scores)

    # ── Hold-out test set ─────────────────────────────────
    print("\n" + "─" * 60)
    print("  Hold-out Test Set Results")
    print("─" * 60)

    mu, sigma  = fit_zscore(train, used_cols)
    z_tr_hand  = zscore_matrix(train, used_cols, mu, sigma)
    z_te       = zscore_matrix(test,  used_cols, mu, sigma)
    s_tr_hand  = baseline_score(z_tr_hand, used_cols, None)
    s_tr_unit  = baseline_score(z_tr_hand, used_cols, unit_w)
    thresh_hand = float(np.quantile(s_tr_hand, TOP_QUANTILE))
    thresh_unit = float(np.quantile(s_tr_unit, TOP_QUANTILE))
    pred_hand   = (baseline_score(z_te, used_cols, None) >= thresh_hand).astype(int)
    pred_unit   = (baseline_score(z_te, used_cols, unit_w) >= thresh_unit).astype(int)

    holdout_rows = []
    for label, pred in (("Baseline (hand weights)", pred_hand), ("Baseline (±1 signs)", pred_unit)):
        m = binary_metrics_dict(y_test, pred, None)
        m["model"] = label
        holdout_rows.append(m)

    lr = make_logistic_regression()
    lr.fit(X_train_arr, y_train)
    lr_m = binary_metrics_dict(y_test, lr.predict(X_test_arr), lr.predict_proba(X_test_arr)[:, 1])
    lr_m["model"] = "Logistic Regression"
    holdout_rows.append(lr_m)

    rf = make_random_forest(n_jobs=1)
    rf.fit(X_train_arr, y_train)
    rf_m = binary_metrics_dict(y_test, rf.predict(X_test_arr), rf.predict_proba(X_test_arr)[:, 1])
    rf_m["model"] = "Random Forest"
    holdout_rows.append(rf_m)

    results_df = pd.DataFrame(holdout_rows).set_index("model")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    results_df.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(results_df.round(3).to_string())
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
