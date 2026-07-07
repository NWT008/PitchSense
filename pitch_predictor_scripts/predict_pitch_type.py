"""
predict_pitch_type_v10_late_curve_filter.py

Safe predictor with a late-vertical-break curveball filter.

What this adds:
    Curveballs in the current dataset often start higher in the frame and/or
    keep y relatively stable early, then show a bigger vertical change late.

This script still trusts raw fastball/changeup predictions when they are solid.
The late-break rule is only used as supporting evidence for curveballs.

Run:
    python .\predict_pitch_type_v10_late_curve_filter.py --model ".\classifier_output_pathshape\best_pitch_classifier_pathshape.joblib" --features ".\test001_features_for_prediction.csv" --pitch-id test001
"""

import argparse
import csv
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pitch type with late-curveball filter.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--pitch-id", default=None)

    parser.add_argument("--trust-confidence", type=float, default=0.60)
    parser.add_argument("--trust-margin", type=float, default=0.12)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--uncertain-margin", type=float, default=0.10)

    parser.add_argument("--no-adjust", action="store_true")
    return parser.parse_args()


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_features(path, pitch_id=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    df = pd.read_csv(path)

    if pitch_id is not None:
        if "pitch_id" not in df.columns:
            raise ValueError("Feature CSV does not contain pitch_id column.")
        df = df[df["pitch_id"].astype(str) == str(pitch_id)].copy()
        if df.empty:
            raise ValueError(f"No row found for pitch_id={pitch_id}")

    return df


def get_expected_features(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        raw_features = []

        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                raw_features.extend(list(cols))
            else:
                raw_features.append(cols)

        return raw_features
    except Exception:
        return None


def prepare_input(df, model):
    expected_features = get_expected_features(model)

    if expected_features is None:
        return df

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(
            "Feature CSV is missing columns expected by model:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    return df[expected_features].copy()


def num(row, col, default=np.nan):
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def read_tracked_points_from_ball_path(path):
    if path is None:
        return []

    path = str(path)

    if path.strip() == "" or not os.path.exists(path):
        return []

    points = []

    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tracked = str(row.get("tracked", row.get("visible", "0"))).strip()
                if tracked not in ["1", "1.0", "true", "True", "yes", "Yes"]:
                    continue

                frame = float(row["frame"])
                x = float(row["x"])
                y = float(row["y"])
                points.append((frame, x, y))
    except Exception:
        return []

    points.sort(key=lambda p: p[0])
    return points


def segment_delta(points):
    if len(points) < 2:
        return 0.0, 0.0

    x0, y0 = points[0][1], points[0][2]
    x1, y1 = points[-1][1], points[-1][2]

    return x1 - x0, y1 - y0


def compute_pathshape_from_points(points):
    if len(points) < 6:
        return {}

    n = len(points)
    a = points[: n // 3]
    b = points[n // 3 : 2 * n // 3]
    c = points[2 * n // 3 :]

    early_dx, early_dy = segment_delta(a)
    middle_dx, middle_dy = segment_delta(b)
    late_dx, late_dy = segment_delta(c)

    xs = np.array([p[1] for p in points], dtype=float)
    ys = np.array([p[2] for p in points], dtype=float)

    start_x = float(xs[0])
    start_y = float(ys[0])
    end_x = float(xs[-1])
    end_y = float(ys[-1])

    abs_net_dx = abs(end_x - start_x)
    abs_net_dy = abs(end_y - start_y)
    total = math.hypot(abs_net_dx, abs_net_dy)

    early_abs_dy = abs(early_dy)
    middle_abs_dy = abs(middle_dy)
    late_abs_dy = abs(late_dy)

    late_ratio = late_abs_dy / (((early_abs_dy + middle_abs_dy) / 2.0) + 1e-9)
    late_share = late_abs_dy / (early_abs_dy + middle_abs_dy + late_abs_dy + 1e-9)

    return {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "abs_net_dx_px": abs_net_dx,
        "abs_net_dy_px": abs_net_dy,
        "total_movement_px": total,
        "early_dy_px": early_dy,
        "middle_dy_px": middle_dy,
        "late_dy_px": late_dy,
        "early_abs_dy_px": early_abs_dy,
        "middle_abs_dy_px": middle_abs_dy,
        "late_abs_dy_px": late_abs_dy,
        "late_vs_early_mid_absdy_ratio": late_ratio,
        "late_dy_share": late_share,
    }


def get_pathshape(row):
    # Prefer existing path-shape columns if present.
    values = {
        "start_y": num(row, "start_y"),
        "abs_net_dx_px": num(row, "abs_net_dx_px"),
        "abs_net_dy_px": num(row, "abs_net_dy_px"),
        "total_movement_px": num(row, "total_movement_px"),
        "early_abs_dy_px": num(row, "early_abs_dy_px"),
        "middle_abs_dy_px": num(row, "middle_abs_dy_px"),
        "late_abs_dy_px": num(row, "late_abs_dy_px"),
        "max_perpendicular_deviation_px": num(row, "max_perpendicular_deviation_px"),
        "mean_abs_perpendicular_deviation_px": num(row, "mean_abs_perpendicular_deviation_px"),
        "quadratic_curvature_y_of_x": abs(num(row, "quadratic_curvature_y_of_x", 0.0)),
        "straightness_ratio": num(row, "straightness_ratio"),
        "avg_speed_px_per_frame": num(row, "avg_speed_px_per_frame"),
    }

    # If early/middle/late are missing because the pipeline created blank columns,
    # compute them from ball_path_file.
    missing_segment_values = (
        pd.isna(values["early_abs_dy_px"])
        or pd.isna(values["middle_abs_dy_px"])
        or pd.isna(values["late_abs_dy_px"])
    )

    if missing_segment_values:
        points = read_tracked_points_from_ball_path(row.get("ball_path_file", ""))
        extra = compute_pathshape_from_points(points)
        for k, v in extra.items():
            values[k] = v

    early = values.get("early_abs_dy_px", np.nan)
    middle = values.get("middle_abs_dy_px", np.nan)
    late = values.get("late_abs_dy_px", np.nan)

    if not pd.isna(early) and not pd.isna(middle) and not pd.isna(late):
        values["late_vs_early_mid_absdy_ratio"] = late / (((early + middle) / 2.0) + 1e-9)
        values["late_dy_share"] = late / (early + middle + late + 1e-9)
    else:
        values["late_vs_early_mid_absdy_ratio"] = np.nan
        values["late_dy_share"] = np.nan

    return values


def top_two(classes, probs):
    order = np.argsort(probs)[::-1]
    top_idx = order[0]
    second_idx = order[1] if len(order) > 1 else order[0]

    return {
        "top_label": classes[top_idx],
        "top_prob": float(probs[top_idx]),
        "second_label": classes[second_idx],
        "second_prob": float(probs[second_idx]),
        "margin": float(probs[top_idx] - probs[second_idx]),
    }


def safe_adjust(classes, raw_probs, row, trust_confidence, trust_margin):
    classes = list(classes)
    probs = np.array(raw_probs, dtype=float).copy()

    needed = {"fastball", "curveball", "changeup"}
    if not needed.issubset(set(classes)):
        return probs, "no adjustment: expected classes not found", get_pathshape(row)

    fast_i = classes.index("fastball")
    curve_i = classes.index("curveball")
    change_i = classes.index("changeup")

    info = top_two(classes, probs)
    raw_label = info["top_label"]
    raw_conf = info["top_prob"]
    raw_margin = info["margin"]
    raw_is_trustworthy = raw_conf >= trust_confidence and raw_margin >= trust_margin

    ps = get_pathshape(row)

    abs_dx = ps["abs_net_dx_px"]
    abs_dy = ps["abs_net_dy_px"]
    total = ps["total_movement_px"]
    start_y = ps["start_y"]
    early_abs = ps["early_abs_dy_px"]
    middle_abs = ps["middle_abs_dy_px"]
    late_abs = ps["late_abs_dy_px"]
    late_ratio = ps["late_vs_early_mid_absdy_ratio"]
    late_share = ps["late_dy_share"]
    max_dev = ps["max_perpendicular_deviation_px"]
    mean_dev = ps["mean_abs_perpendicular_deviation_px"]
    curvature = ps["quadratic_curvature_y_of_x"]

    curve_prob = float(probs[curve_i])

    # Dataset-supported curveball behavior:
    # curveballs start higher on average and have more late vertical movement.
    # But not every curveball follows it, so this is supporting evidence only.
    starts_high = (not pd.isna(start_y)) and start_y <= 825

    late_vertical_break = (
        not pd.isna(late_abs)
        and not pd.isna(late_ratio)
        and not pd.isna(late_share)
        and late_abs >= 60
        and late_ratio >= 1.45
        and late_share >= 0.42
    )

    huge_late_vertical_break = (
        not pd.isna(late_abs)
        and not pd.isna(late_ratio)
        and late_abs >= 95
        and late_ratio >= 1.70
    )

    combined_curve_movement = (
        ((abs_dx >= 260 and abs_dy >= 150) or total >= 385)
        and (max_dev >= 70 or mean_dev >= 35 or curvature >= 0.035)
    )

    very_strong_curve_evidence = combined_curve_movement or (late_vertical_break and starts_high) or huge_late_vertical_break

    # Trust solid raw fastball/changeup unless curveball evidence is very strong.
    if raw_label in ["fastball", "changeup"] and raw_is_trustworthy and not very_strong_curve_evidence:
        return probs, f"trusted raw {raw_label}: no strong late-curve evidence", ps

    # Boost curveball only if the curve probability is already plausible OR evidence is huge.
    if very_strong_curve_evidence and (curve_prob >= 0.18 or raw_label == "curveball"):
        probs[curve_i] *= 1.65
        probs[fast_i] *= 0.85
        probs[change_i] *= 0.90

        reasons = []
        if starts_high:
            reasons.append("starts higher in frame")
        if late_vertical_break:
            reasons.append("late vertical break")
        if huge_late_vertical_break:
            reasons.append("huge late vertical break")
        if combined_curve_movement:
            reasons.append("large combined movement/shape")

        reason = "boosted curveball: " + ", ".join(reasons)

    elif raw_label == "curveball" and not very_strong_curve_evidence:
        probs[curve_i] *= 0.70

        if info["second_label"] == "fastball":
            probs[fast_i] *= 1.18
            alt = "fastball"
        elif info["second_label"] == "changeup":
            probs[change_i] *= 1.18
            alt = "changeup"
        else:
            probs[fast_i] *= 1.08
            probs[change_i] *= 1.08
            alt = "fastball/changeup"

        reason = f"softened curveball: no strong late-curve evidence, boosted {alt}"

    else:
        reason = "model probabilities mostly kept"

    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob

    return probs, reason, ps


def decide_label(classes, probs, min_confidence, uncertain_margin):
    info = top_two(classes, probs)

    if info["top_prob"] < min_confidence:
        return (
            f"uncertain, leaning {info['top_label']}",
            info["top_prob"],
            f"top probability {info['top_prob']:.3f} < {min_confidence:.3f}",
        )

    if info["margin"] < uncertain_margin:
        return (
            f"uncertain, between {info['top_label']} and {info['second_label']}",
            info["top_prob"],
            f"top-two margin {info['margin']:.3f} < {uncertain_margin:.3f}",
        )

    return info["top_label"], info["top_prob"], "confident enough"


def print_probs(title, classes, probs):
    print(title)
    pairs = list(zip(classes, probs))
    pairs.sort(key=lambda item: item[1], reverse=True)
    for label, prob in pairs:
        print(f"  {label}: {float(prob):.3f}")


def main():
    args = parse_args()

    model = load_model(args.model)
    df = load_features(args.features, pitch_id=args.pitch_id)
    X = prepare_input(df, model)

    raw_labels = model.predict(X)

    if not hasattr(model, "predict_proba"):
        for label in raw_labels:
            print(f"Predicted pitch type: {label}")
        return

    raw_probs = model.predict_proba(X)
    classes = list(model.classes_)

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        probs_i = raw_probs[i]
        raw_label = raw_labels[i]

        if args.no_adjust:
            final_probs = probs_i
            reason = "adjustment disabled"
            ps = get_pathshape(row)
        else:
            final_probs, reason, ps = safe_adjust(
                classes,
                probs_i,
                row,
                args.trust_confidence,
                args.trust_margin,
            )

        final_label, confidence, note = decide_label(
            classes,
            final_probs,
            args.min_confidence,
            args.uncertain_margin,
        )

        print("\n[PITCH PREDICTION]")
        if "pitch_id" in row:
            print(f"Pitch ID: {row.get('pitch_id')}")
        if "pitcher_handedness" in row:
            print(f"Pitcher handedness: {row.get('pitcher_handedness')}")

        print(f"Raw predicted pitch type: {raw_label}")
        print(f"Final predicted pitch type: {final_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Adjustment: {reason}")
        print(f"Decision note: {note}")

        print("\nLate-curve evidence:")
        print(f"  start_y: {ps.get('start_y')}")
        print(f"  early_abs_dy_px: {ps.get('early_abs_dy_px')}")
        print(f"  middle_abs_dy_px: {ps.get('middle_abs_dy_px')}")
        print(f"  late_abs_dy_px: {ps.get('late_abs_dy_px')}")
        print(f"  late_vs_early_mid_absdy_ratio: {ps.get('late_vs_early_mid_absdy_ratio')}")
        print(f"  late_dy_share: {ps.get('late_dy_share')}")

        print("\nOther key movement features:")
        print(f"  abs_net_dx_px: {ps.get('abs_net_dx_px')}")
        print(f"  abs_net_dy_px: {ps.get('abs_net_dy_px')}")
        print(f"  total_movement_px: {ps.get('total_movement_px')}")
        print(f"  max_perpendicular_deviation_px: {ps.get('max_perpendicular_deviation_px')}")
        print(f"  mean_abs_perpendicular_deviation_px: {ps.get('mean_abs_perpendicular_deviation_px')}")
        print(f"  quadratic_curvature_y_of_x_abs: {ps.get('quadratic_curvature_y_of_x')}")

        print_probs("\nRaw probability distribution:", classes, probs_i)
        print_probs("\nFinal probability distribution:", classes, final_probs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
