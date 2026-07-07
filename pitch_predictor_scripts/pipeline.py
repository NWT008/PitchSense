"""
pipeline.py

End-to-end GPITCHU runner.

This script:
    1. runs ball_tracker_model.py
    2. uses manual anchor frames normally
    3. creates temporary tracker files
    4. builds a feature row matching compiled_pitch_features_30samples.csv
    5. runs predict_pitch_type.py
    6. prints the final predicted pitch type and probabilities
    7. deletes temporary tracker files unless --keep-files is used

Important:
    Tracker output is visible by default so anchor clicking works clearly.
    Use --quiet-tracker only if you do not want tracker messages printed.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GPITCHU tracker and predictor."
    )

    parser.add_argument("--video", required=True, help="Input pitch video path.")
    parser.add_argument("--pitch-id", required=True, help="Unique pitch ID, for example test001.")
    parser.add_argument("--pitch-type", default="unknown", help="Known type if available. Usually unknown for test clips.")

    parser.add_argument(
        "--pitcher-handedness",
        choices=["RHP", "LHP", "rhp", "lhp"],
        default=None,
        help="Pitcher handedness. If omitted, the script asks for RHP or LHP.",
    )

    parser.add_argument("--video-angle-id", default="unknown")

    parser.add_argument("--tracker", default="ball_tracker_model.py")
    parser.add_argument("--predictor", default="predict_pitch_type.py")
    parser.add_argument("--model", default=os.path.join("classifier_output", "best_pitch_classifier.joblib"))
    parser.add_argument("--training-csv", default="compiled_pitch_features_30samples.csv")
    parser.add_argument("--thresholds", default=os.path.join("classifier_output", "balanced_thresholds.json"))

    parser.add_argument("--roi", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--manual-anchor-frames", nargs="+", type=int, required=True)

    parser.add_argument("--pitch-direction", default="right", choices=["right", "left"])
    parser.add_argument("--corridor-radius", type=float, default=70.0)
    parser.add_argument("--search-radius", type=float, default=60.0)

    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-debug", action="store_true")

    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep temporary tracker outputs in a saved folder for debugging.",
    )

    parser.add_argument(
        "--quiet-tracker",
        action="store_true",
        help="Hide tracker output. Not recommended when using manual anchors.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all command details.",
    )

    return parser.parse_args()


def normalize_handedness(value):
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ["R", "RH", "RHP", "RIGHT", "RIGHTY", "RIGHT-HANDED"]:
        return "RHP"
    if s in ["L", "LH", "LHP", "LEFT", "LEFTY", "LEFT-HANDED"]:
        return "LHP"
    return None


def ask_handedness():
    while True:
        ans = input("Pitcher handedness? Type RHP or LHP: ").strip()
        norm = normalize_handedness(ans)
        if norm:
            return norm
        print("Please enter either RHP or LHP.")


def run_command(cmd, label, show_output=False, show_command=False):
    print("\n" + "=" * 80)
    print(f"[RUNNING] {label}")
    print("=" * 80)

    if show_command:
        print(" ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
        print()

    if show_output:
        result = subprocess.run(cmd)
        output = ""
    else:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout or ""

    if result.returncode != 0:
        if output:
            print("\n[CAPTURED OUTPUT]")
            print(output)
        raise RuntimeError(f"{label} failed with return code {result.returncode}")

    print(f"[DONE] {label}")

    return output


def to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_ball_path(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compute_features_from_ball_path(rows):
    tracked_points = []
    anchor_count = 0
    total_rows = 0

    for row in rows:
        total_rows += 1

        tracked_raw = str(row.get("tracked", row.get("visible", "0"))).strip()
        tracked = tracked_raw in ["1", "1.0", "true", "True", "yes", "Yes"]

        anchor_raw = str(row.get("anchor", "0")).strip()
        anchor = anchor_raw in ["1", "1.0", "true", "True", "yes", "Yes"]
        if anchor:
            anchor_count += 1

        frame = to_float(row.get("frame"))
        x = to_float(row.get("x"))
        y = to_float(row.get("y"))

        if tracked and frame is not None and x is not None and y is not None:
            tracked_points.append((frame, x, y))

    tracked_points.sort(key=lambda item: item[0])

    if len(tracked_points) < 2:
        return {
            "status": "not_enough_points",
            "num_points": len(tracked_points),
            "tracked_fraction": len(tracked_points) / total_rows if total_rows else 0.0,
            "anchor_count": anchor_count,
        }

    frames = [p[0] for p in tracked_points]
    xs = [p[1] for p in tracked_points]
    ys = [p[2] for p in tracked_points]

    dists = []
    speeds = []

    for i in range(1, len(tracked_points)):
        frame_delta = max(frames[i] - frames[i - 1], 1.0)
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = math.hypot(dx, dy)
        dists.append(dist)
        speeds.append(dist / frame_delta)

    path_length = sum(dists)
    direct_distance = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    straightness_ratio = direct_distance / path_length if path_length > 1e-9 else 0.0

    ax, ay = xs[0], ys[0]
    bx, by = xs[-1], ys[-1]
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby

    signed_devs = []
    abs_devs = []

    for x, y in zip(xs, ys):
        if ab2 < 1e-9:
            signed_dev = 0.0
            abs_dev = 0.0
        else:
            signed_dev = (abx * (y - ay) - aby * (x - ax)) / math.sqrt(ab2)
            t = ((x - ax) * abx + (y - ay) * aby) / ab2
            t = max(0.0, min(1.0, t))
            qx = ax + t * abx
            qy = ay + t * aby
            abs_dev = math.hypot(x - qx, y - qy)

        signed_devs.append(signed_dev)
        abs_devs.append(abs_dev)

    curvature = ""
    try:
        n = len(xs)
        sx0 = n
        sx1 = sum(xs)
        sx2 = sum(x * x for x in xs)
        sx3 = sum(x * x * x for x in xs)
        sx4 = sum(x * x * x * x for x in xs)
        sy0 = sum(ys)
        sy1 = sum(x * y for x, y in zip(xs, ys))
        sy2 = sum(x * x * y for x, y in zip(xs, ys))

        A = [[sx4, sx3, sx2], [sx3, sx2, sx1], [sx2, sx1, sx0]]
        B = [sy2, sy1, sy0]

        def det3(M):
            return (
                M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
            )

        D = det3(A)
        if abs(D) > 1e-12:
            A0 = [[B[i] if j == 0 else A[i][j] for j in range(3)] for i in range(3)]
            curvature = det3(A0) / D
    except Exception:
        curvature = ""

    duration_frames = frames[-1] - frames[0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0
    speed_std = math.sqrt(sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)) if speeds else 0.0

    net_dx = xs[-1] - xs[0]
    net_dy = ys[-1] - ys[0]
    avg_vx = net_dx / max(duration_frames, 1.0)
    avg_vy = net_dy / max(duration_frames, 1.0)

    return {
        "status": "ok",
        "num_points": len(tracked_points),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "duration_frames": int(duration_frames),
        "tracked_fraction": len(tracked_points) / total_rows if total_rows else 0.0,
        "anchor_count": anchor_count,
        "avg_speed_px_per_frame": avg_speed,
        "max_speed_px_per_frame": max_speed,
        "speed_std": speed_std,
        "path_length_px": path_length,
        "direct_distance_px": direct_distance,
        "straightness_ratio": straightness_ratio,
        "max_perpendicular_deviation_px": max(abs_devs) if abs_devs else 0.0,
        "mean_abs_perpendicular_deviation_px": sum(abs(v) for v in signed_devs) / len(signed_devs) if signed_devs else 0.0,
        "max_signed_perpendicular_deviation_px": max(signed_devs) if signed_devs else 0.0,
        "min_signed_perpendicular_deviation_px": min(signed_devs) if signed_devs else 0.0,
        "quadratic_curvature_y_of_x": curvature,
        "net_dx_px": net_dx,
        "net_dy_px": net_dy,
        "avg_vx_px_per_frame": avg_vx,
        "avg_vy_px_per_frame": avg_vy,
        "abs_net_dx_px": abs(net_dx),
        "abs_net_dy_px": abs(net_dy),
        "abs_avg_vx_px_per_frame": abs(avg_vx),
        "abs_avg_vy_px_per_frame": abs(avg_vy),
    }


def build_compatible_feature_csv(training_csv, output_csv, pitch_id, pitch_type, handedness, ball_path_file, video_angle_id, feature_values):
    training_df = pd.read_csv(training_csv)
    columns = list(training_df.columns)
    row = {col: "" for col in columns}

    row["pitch_id"] = pitch_id
    row["pitch_type"] = pitch_type
    row["pitcher_handedness"] = handedness
    row["ball_path_file"] = str(ball_path_file)

    if "video_angle_id" in row:
        row["video_angle_id"] = video_angle_id

    if "notes" in row:
        row["notes"] = "Temporary feature row generated by pipeline.py"

    for key, value in feature_values.items():
        if key in row:
            row[key] = value

    pd.DataFrame([row], columns=columns).to_csv(output_csv, index=False)
    return output_csv


def main():
    args = parse_args()

    handedness = normalize_handedness(args.pitcher_handedness)
    if handedness is None:
        handedness = ask_handedness()

    tracker_path = Path(args.tracker)
    predictor_path = Path(args.predictor)
    model_path = Path(args.model)
    training_csv = Path(args.training_csv)
    thresholds_path = Path(args.thresholds) if args.thresholds else None

    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker script not found: {tracker_path}")
    if not predictor_path.exists():
        raise FileNotFoundError(f"Predictor script not found: {predictor_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")

    if args.keep_files:
        temp_root = Path(f"{args.pitch_id}_debug_outputs")
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_context = None
        work_dir = temp_root
        print(f"[INFO] Keeping intermediate files in: {work_dir}")
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="gpitchu_")
        work_dir = Path(temp_context.name)

    try:
        tracked_video = work_dir / f"{args.pitch_id}_tracked.mp4"
        ball_path_csv = work_dir / f"{args.pitch_id}_ball_path.csv"
        metrics_txt = work_dir / f"{args.pitch_id}_metrics.txt"
        compatible_features_csv = work_dir / f"{args.pitch_id}_features_for_prediction.csv"

        tracker_cmd = [
            sys.executable,
            str(tracker_path),
            "--video",
            str(args.video),
            "--roi",
            *[str(v) for v in args.roi],
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--manual-anchor-frames",
            *[str(v) for v in args.manual_anchor_frames],
            "--pitch-direction",
            args.pitch_direction,
            "--corridor-radius",
            str(args.corridor_radius),
            "--search-radius",
            str(args.search_radius),
            "--pitch-id",
            args.pitch_id,
            "--pitch-type",
            args.pitch_type,
            "--output",
            str(tracked_video),
            "--csv",
            str(ball_path_csv),
            "--metrics",
            str(metrics_txt),
        ]

        if not args.no_display:
            tracker_cmd.append("--display")
        if not args.no_debug:
            tracker_cmd.append("--debug")

        # Show tracker output by default because manual anchors need visible instructions.
        run_command(
            tracker_cmd,
            "Track pitch",
            show_output=(not args.quiet_tracker),
            show_command=args.verbose,
        )

        if not ball_path_csv.exists():
            raise FileNotFoundError(f"Ball path CSV not found after tracker run: {ball_path_csv}")

        print("\n[INFO] Computing feature row from tracker path...")
        rows = read_ball_path(ball_path_csv)
        feature_values = compute_features_from_ball_path(rows)

        build_compatible_feature_csv(
            training_csv=training_csv,
            output_csv=compatible_features_csv,
            pitch_id=args.pitch_id,
            pitch_type=args.pitch_type,
            handedness=handedness,
            ball_path_file=ball_path_csv,
            video_angle_id=args.video_angle_id,
            feature_values=feature_values,
        )

        predictor_cmd = [
            sys.executable,
            str(predictor_path),
            "--model",
            str(model_path),
            "--features",
            str(compatible_features_csv),
            "--pitch-id",
            args.pitch_id,
        ]

        if thresholds_path is not None and thresholds_path.exists():
            predictor_cmd.extend(["--thresholds", str(thresholds_path)])

        predictor_output = run_command(
            predictor_cmd,
            "Predict pitch type",
            show_output=False,
            show_command=args.verbose,
        )

        if predictor_output:
            print(predictor_output.strip())

        print("\n" + "=" * 80)
        print("[PIPELINE COMPLETE]")
        print("=" * 80)
        print(f"Handedness used: {handedness}")

        if args.keep_files:
            print(f"Intermediate files kept in: {work_dir}")
        else:
            print("Intermediate tracker files were deleted.")

    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
