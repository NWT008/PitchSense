"""
train_pitch_classifier_pathshape.py

Trains GPITCHU pitch classifier using enhanced path-shape features.

Run:
    python train_pitch_classifier_pathshape.py --features compiled_pitch_features_pathshape_30samples.csv --out-dir classifier_output_pathshape
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def parse_args():
    p = argparse.ArgumentParser(description="Train enhanced path-shape pitch classifier.")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", default="classifier_output_pathshape")
    p.add_argument("--target", default="pitch_type")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def get_feature_columns(df, target):
    exclude = {target, "pitch_id", "ball_path_file", "source_file", "status", "notes"}
    numeric = []
    categorical = []

    # Smaller, stronger feature set to avoid overfitting 30 rows.
    preferred_numeric = [
        "avg_speed_px_per_frame", "speed_std",
        "abs_net_dx_px", "abs_net_dy_px", "total_movement_px",
        "vertical_to_horizontal_ratio", "horizontal_to_vertical_ratio",
        "max_perpendicular_deviation_px", "mean_abs_perpendicular_deviation_px",
        "quadratic_curvature_y_of_x", "abs_quadratic_curvature", "straightness_ratio",
        "early_dx_px", "middle_dx_px", "late_dx_px",
        "early_dy_px", "middle_dy_px", "late_dy_px",
        "early_to_late_dx_change_px", "early_to_late_dy_change_px", "slope_change_late_minus_early",
        "horizontal_return_px", "horizontal_return_ratio",
        "arm_side_net_dx_px", "arm_side_return_px", "late_arm_side_dx_px",
        "pattern_curveball_score", "pattern_fastball_score", "pattern_changeup_score",
    ]

    preferred_categorical = [
        "pitcher_handedness", "movement_category", "horizontal_category", "vertical_category",
        "shape_category", "horizontal_return_category", "arm_side_return_category",
    ]

    for c in preferred_numeric:
        if c in df.columns:
            numeric.append(c)
    for c in preferred_categorical:
        if c in df.columns:
            categorical.append(c)

    return numeric, categorical


def build_preprocessor(numeric, categorical):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers = [("num", num_pipe, numeric)]

    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical))

    return ColumnTransformer(transformers)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.features)
    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")

    df = df[df[args.target].notna()].copy()
    y = df[args.target].astype(str).str.lower()
    numeric, categorical = get_feature_columns(df, args.target)
    X = df[numeric + categorical].copy()

    preprocessor = build_preprocessor(numeric, categorical)

    models = {
        "svm_pathshape": SVC(C=0.8, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=args.random_state),
        "random_forest_pathshape": RandomForestClassifier(n_estimators=300, max_depth=3, class_weight="balanced", random_state=args.random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    results = []
    reports = {}

    print("Class counts:")
    print(y.value_counts().to_string())

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        pred = cross_val_predict(pipe, X, y, cv=cv)
        acc = accuracy_score(y, pred)
        results.append({"model": name, "mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std()), "cross_val_predict_accuracy": float(acc)})
        reports[name] = {
            "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=sorted(y.unique())).tolist(),
            "labels": sorted(y.unique()),
        }
        print(f"\n=== {name} ===")
        print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(classification_report(y, pred, zero_division=0))

    results_df = pd.DataFrame(results).sort_values("mean_accuracy", ascending=False)
    results_path = os.path.join(args.out_dir, "model_cv_results_pathshape.csv")
    results_df.to_csv(results_path, index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    best_pipe = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    best_pipe.fit(X, y)

    model_path = os.path.join(args.out_dir, "best_pitch_classifier_pathshape.joblib")
    joblib.dump(best_pipe, model_path)

    metadata = {
        "best_model": best_name,
        "target": args.target,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "classes": sorted(y.unique()),
        "num_samples": int(len(df)),
        "class_counts": y.value_counts().to_dict(),
        "note": "Path-shape classifier using early/middle/late movement and horizontal-return features.",
    }
    with open(os.path.join(args.out_dir, "model_metadata_pathshape.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.out_dir, "classification_reports_pathshape.json"), "w") as f:
        json.dump(reports, f, indent=2)

    print("\nSaved:")
    print(model_path)
    print(results_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
