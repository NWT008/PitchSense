import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class Candidate:
    x: float
    y: float
    area: float
    radius: float
    brightness: float
    motion: float
    circularity: float
    score: float


def parse_args():
    p = argparse.ArgumentParser(description="Anchor-guided full-path baseball tracker")

    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", default="tracked_pitch_v10.mp4", help="Output video path")
    p.add_argument("--csv", default="ball_path_v10.csv", help="Output CSV path")
    p.add_argument("--metrics", default="pitch_metrics_v10.txt", help="Output metrics path")

    p.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                   help="ROI in full-frame coordinates")
    p.add_argument("--select-roi", action="store_true", help="Interactively select ROI")
    p.add_argument("--start-frame", type=int, default=0, help="First frame to process")
    p.add_argument("--end-frame", type=int, default=-1, help="Last frame to process inclusive, -1 means video end")

    p.add_argument("--manual-anchor-frames", nargs="*", type=int, default=[],
                   help="Frame numbers where user will click the actual baseball")
    p.add_argument("--anchor-file", default=None,
                   help="Optional CSV with columns frame,x,y. If given, manual clicks are loaded from file.")

    p.add_argument("--pitch-direction", choices=["right", "left"], default="right",
                   help="Expected horizontal direction of ball travel")
    p.add_argument("--corridor-radius", type=float, default=70.0,
                   help="Allowed pixel distance from interpolated anchor path")
    p.add_argument("--search-radius", type=float, default=60.0,
                   help="Allowed pixel distance from predicted point during forward tracking")
    p.add_argument("--max-jump", type=float, default=120.0,
                   help="Maximum allowed jump between consecutive detections")
    p.add_argument("--min-area", type=float, default=2.0)
    p.add_argument("--max-area", type=float, default=450.0)
    p.add_argument("--motion-thresh", type=int, default=10)
    p.add_argument("--bright-v", type=int, default=105)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--display", action="store_true")
    p.add_argument("--resize-display", type=float, default=0.65)
    p.add_argument("--dataset-dir", default="pitch_dataset",
                   help="Dataset folder where labels, crops, paths, and features are saved")
    p.add_argument("--save-dataset", action="store_true",
                   help="Save frame labels, crops, trajectory path, and feature rows for training")
    p.add_argument("--pitch-id", default=None,
                   help="Unique pitch ID. Defaults to video filename without extension")
    p.add_argument("--pitch-type", default="unknown",
                   help="Optional pitch label such as fastball, curveball, slider, changeup")
    p.add_argument("--crop-size", type=int, default=96,
                   help="Square crop size around tracked ball for detector training")
    p.add_argument("--features-csv", default=None,
                   help="Optional global features CSV. Defaults to dataset-dir/features.csv")
    p.add_argument("--manifest-csv", default=None,
                   help="Optional crop/label manifest CSV. Defaults to dataset-dir/train_manifest.csv")

    return p.parse_args()


def read_video(path: str, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0 or end_frame >= total:
        end_frame = total - 1

    frames = []
    frame_numbers = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fno = start_frame
    while fno <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        frame_numbers.append(fno)
        fno += 1

    cap.release()
    return frames, frame_numbers, fps, total


def choose_roi(frame):
    r = cv2.selectROI("Select ROI, then press ENTER", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI, then press ENTER")
    x, y, w, h = map(int, r)
    return (x, y, x + w, y + h)


def crop_roi(frame, roi):
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2].copy()


def click_anchor(frame, frame_no: int, roi=None, resize=0.8):
    shown = frame.copy()
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(shown, (x1, y1), (x2, y2), (255, 0, 255), 2)

    disp = cv2.resize(shown, None, fx=resize, fy=resize)
    clicked = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x / resize, y / resize))
            temp = disp.copy()
            cv2.circle(temp, (x, y), 6, (0, 255, 255), -1)
            cv2.imshow(f"Click ball frame {frame_no}", temp)

    name = f"Click ball frame {frame_no}"
    cv2.imshow(name, disp)
    cv2.setMouseCallback(name, on_mouse)

    print(f"[CLICK] Frame {frame_no}: click the baseball, then press ENTER. Press 's' to skip.")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in [13, 10]:
            break
        if k == ord("s"):
            clicked.clear()
            break
        if k == 27:
            clicked.clear()
            break

    cv2.destroyWindow(name)

    if clicked:
        x, y = clicked[0]
        return (float(x), float(y))
    return None


def load_anchor_file(path: str) -> Dict[int, Point]:
    anchors = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            anchors[frame] = (float(row["x"]), float(row["y"]))
    return anchors


def save_anchor_file(path: str, anchors: Dict[int, Point]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])
        for frame in sorted(anchors):
            x, y = anchors[frame]
            writer.writerow([frame, x, y])


def build_evidence_maps(roi_frames, motion_thresh):
    grays = []
    hsvs = []
    for fr in roi_frames:
        blur = cv2.GaussianBlur(fr, (5, 5), 0)
        grays.append(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))
        hsvs.append(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV))

    motion_maps = []
    for i in range(len(grays)):
        if i == 0:
            diff = np.zeros_like(grays[i])
        else:
            d1 = cv2.absdiff(grays[i], grays[i - 1])
            if i >= 2:
                d2 = cv2.absdiff(grays[i], grays[i - 2])
                diff = cv2.max(d1, d2)
            else:
                diff = d1
        _, mot = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
        mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mot = cv2.dilate(mot, np.ones((3, 3), np.uint8), iterations=1)
        motion_maps.append(mot)

    return grays, hsvs, motion_maps


def point_line_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def interp_anchor_path(frame_no: int, anchors: Dict[int, Point]) -> Optional[Point]:
    keys = sorted(anchors)
    if not keys:
        return None
    if frame_no in anchors:
        return anchors[frame_no]
    if frame_no < keys[0] or frame_no > keys[-1]:
        return None

    lo = None
    hi = None
    for k in keys:
        if k < frame_no:
            lo = k
        elif k > frame_no:
            hi = k
            break

    if lo is None or hi is None:
        return None

    t = (frame_no - lo) / (hi - lo)
    x = anchors[lo][0] * (1 - t) + anchors[hi][0] * t
    y = anchors[lo][1] * (1 - t) + anchors[hi][1] * t
    return (x, y)


def segment_for_frame(frame_no: int, anchors: Dict[int, Point]):
    keys = sorted(anchors)
    if len(keys) < 2:
        return None
    for a, b in zip(keys[:-1], keys[1:]):
        if a <= frame_no <= b:
            return anchors[a], anchors[b]
    return None


def extract_candidates(frame_bgr, gray, hsv, motion_map, args, expected=None, segment=None):
    h, w = gray.shape[:2]
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Evidence masks
    bright_mask = ((v >= args.bright_v) & (s <= 170)).astype(np.uint8) * 255

    # Difference of Gaussian for tiny bright/dark blobs
    g1 = cv2.GaussianBlur(gray, (3, 3), 0)
    g2 = cv2.GaussianBlur(gray, (11, 11), 0)
    dog = cv2.absdiff(g1, g2)
    _, dog_mask = cv2.threshold(dog, 8, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(motion_map, dog_mask)
    mask = cv2.bitwise_and(mask, cv2.bitwise_or(bright_mask, dog_mask))

    if expected is not None:
        ex, ey = expected
        restrict = np.zeros_like(mask)
        cv2.circle(restrict, (int(ex), int(ey)), int(args.search_radius), 255, -1)
        mask = cv2.bitwise_and(mask, restrict)

    if segment is not None:
        a, b = segment
        corridor = np.zeros_like(mask)
        # Draw a thick line corridor between anchor positions
        cv2.line(corridor, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 255, int(args.corridor_radius * 2))
        mask = cv2.bitwise_and(mask, corridor)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < args.min_area or area > args.max_area:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        perim = cv2.arcLength(c, True)
        circularity = 0.0
        if perim > 1e-6:
            circularity = 4.0 * math.pi * area / (perim * perim)

        x, y, bw, bh = cv2.boundingRect(c)
        radius = 0.5 * max(bw, bh)

        patch_v = v[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        patch_m = motion_map[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]

        brightness = float(np.mean(patch_v)) if patch_v.size else 0.0
        motion = float(np.mean(patch_m)) if patch_m.size else 0.0

        score = 0.0
        score += min(brightness / 255.0, 1.0) * 2.0
        score += min(motion / 255.0, 1.0) * 2.5
        score += min(circularity, 1.0) * 1.0
        score += max(0.0, 1.0 - abs(area - 35.0) / 120.0) * 1.0

        if expected is not None:
            d = math.hypot(cx - expected[0], cy - expected[1])
            score += max(0.0, 1.0 - d / max(args.search_radius, 1.0)) * 3.0

        if segment is not None:
            dline = point_line_distance((cx, cy), segment[0], segment[1])
            score += max(0.0, 1.0 - dline / max(args.corridor_radius, 1.0)) * 3.0

        cands.append(Candidate(cx, cy, area, radius, brightness, motion, circularity, score))

    cands.sort(key=lambda z: z.score, reverse=True)
    return cands, mask


def direction_ok(prev: Point, cur: Point, direction: str, min_progress=-20.0):
    dx = cur[0] - prev[0]
    if direction == "right":
        return dx >= min_progress
    return dx <= -min_progress


def smooth_path(path: Dict[int, Point], anchors: Dict[int, Point], window=5) -> Dict[int, Point]:
    if not path:
        return path
    keys = sorted(path)
    smoothed = {}

    for k in keys:
        if k in anchors:
            smoothed[k] = anchors[k]
            continue

        xs, ys, ws = [], [], []
        for j in keys:
            if abs(j - k) <= window:
                w = 1.0 / (1.0 + abs(j - k))
                xs.append(path[j][0])
                ys.append(path[j][1])
                ws.append(w)
        if ws:
            sw = sum(ws)
            smoothed[k] = (sum(x * w for x, w in zip(xs, ws)) / sw,
                           sum(y * w for y, w in zip(ys, ws)) / sw)
        else:
            smoothed[k] = path[k]

    return smoothed


def build_path(frame_numbers, roi_frames, grays, hsvs, motion_maps, anchors_global, roi, args):
    x1, y1, _, _ = roi
    anchors = {f: (pt[0] - x1, pt[1] - y1) for f, pt in anchors_global.items()}
    anchors = {f: pt for f, pt in anchors.items() if f in frame_numbers}

    if not anchors:
        print("[WARN] No anchors inside the processed frame window.")
        return {}, {}

    first_anchor = min(anchors)
    last_anchor = max(anchors)

    path_roi: Dict[int, Point] = {}
    debug_candidates = {}

    # Force anchors into path
    for f, pt in anchors.items():
        path_roi[f] = pt

    # Track between anchors using corridor search
    for idx, f in enumerate(frame_numbers):
        if f < first_anchor or f > last_anchor:
            continue

        if f in anchors:
            continue

        local_idx = idx
        expected = interp_anchor_path(f, anchors)
        segment = segment_for_frame(f, anchors)

        cands, _ = extract_candidates(
            roi_frames[local_idx],
            grays[local_idx],
            hsvs[local_idx],
            motion_maps[local_idx],
            args,
            expected=expected,
            segment=segment,
        )
        debug_candidates[f] = cands[:8]

        if cands:
            best = cands[0]
            path_roi[f] = (best.x, best.y)
        elif expected is not None:
            path_roi[f] = expected

    keys = sorted(k for k in path_roi if k <= last_anchor)
    if len(keys) >= 2:
        prev_f, cur_f = keys[-2], keys[-1]
        prev = path_roi[prev_f]
        cur = path_roi[cur_f]
        dt = max(cur_f - prev_f, 1)
        vx = (cur[0] - prev[0]) / dt
        vy = (cur[1] - prev[1]) / dt

        after = [f for f in frame_numbers if f > last_anchor]
        last_point = cur
        last_f = cur_f

        for f in after:
            gap = f - last_f
            pred = (last_point[0] + vx * gap, last_point[1] + vy * gap)

            idx = frame_numbers.index(f)
            cands, _ = extract_candidates(
                roi_frames[idx],
                grays[idx],
                hsvs[idx],
                motion_maps[idx],
                args,
                expected=pred,
                segment=None,
            )
            debug_candidates[f] = cands[:8]

            chosen = None
            for cand in cands:
                cand_pt = (cand.x, cand.y)
                if math.hypot(cand.x - pred[0], cand.y - pred[1]) <= args.search_radius:
                    if direction_ok(last_point, cand_pt, args.pitch_direction):
                        if math.hypot(cand.x - last_point[0], cand.y - last_point[1]) <= args.max_jump:
                            chosen = cand_pt
                            break

            if chosen is None:
                chosen = pred

            h, w = grays[idx].shape[:2]
            if chosen[0] < 0 or chosen[0] >= w or chosen[1] < 0 or chosen[1] >= h:
                break

            path_roi[f] = chosen
            vx = 0.7 * vx + 0.3 * ((chosen[0] - last_point[0]) / max(gap, 1))
            vy = 0.7 * vy + 0.3 * ((chosen[1] - last_point[1]) / max(gap, 1))
            last_point = chosen
            last_f = f

    path_roi = smooth_path(path_roi, anchors, window=3)

    path_global = {f: (pt[0] + x1, pt[1] + y1) for f, pt in path_roi.items()}
    return path_global, debug_candidates


def compute_metrics(path: Dict[int, Point], fps: float, anchors: Dict[int, Point]):
    keys = sorted(path)
    if len(keys) < 2:
        return {"status": "not_enough_points"}

    pts = np.array([path[k] for k in keys], dtype=np.float64)
    frames = np.array(keys, dtype=np.float64)

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    frame_deltas = np.diff(frames)
    speeds = dists / np.maximum(frame_deltas, 1.0)

    start = pts[0]
    end = pts[-1]
    direct = np.linalg.norm(end - start)
    path_len = np.sum(dists)
    straightness = direct / path_len if path_len > 1e-6 else 0.0

    # Distance from line start-end
    a = start
    b = end
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    deviations = []
    for p in pts:
        if ab2 < 1e-6:
            deviations.append(0.0)
        else:
            t = np.clip(float(np.dot(p - a, ab) / ab2), 0, 1)
            q = a + t * ab
            deviations.append(float(np.linalg.norm(p - q)))

    max_dev = max(deviations) if deviations else 0.0

    curvature = 0.0
    try:
        x = pts[:, 0]
        y = pts[:, 1]
        if len(np.unique(np.round(x, 1))) >= 3:
            coeffs = np.polyfit(x, y, 2)
            curvature = float(coeffs[0])
    except Exception:
        curvature = 0.0

    if straightness >= 0.93 and max_dev < 45:
        label = "fastball-like/straight"
    elif max_dev >= 45 or abs(curvature) > 0.00025:
        label = "breaking-ball-like/curved"
    else:
        label = "uncertain"

    return {
        "status": "ok",
        "num_points": len(keys),
        "first_frame": int(keys[0]),
        "last_frame": int(keys[-1]),
        "duration_sec": float((keys[-1] - keys[0]) / fps),
        "avg_speed_px_per_frame": float(np.mean(speeds)) if len(speeds) else 0.0,
        "max_speed_px_per_frame": float(np.max(speeds)) if len(speeds) else 0.0,
        "path_length_px": float(path_len),
        "direct_distance_px": float(direct),
        "straightness_ratio": float(straightness),
        "max_perpendicular_deviation_px": float(max_dev),
        "quadratic_curvature_y_of_x": float(curvature),
        "anchor_count": len(anchors),
        "heuristic_label": label,
        "note": "Heuristic only. Reliable pitch classification requires labeled examples and full trajectory."
    }


def draw_output(frames, frame_numbers, roi, path, anchors, debug_candidates, args, fps):
    x1, y1, x2, y2 = roi
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    trail = []

    for frame, f in zip(frames, frame_numbers):
        vis = frame.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f"frame {f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if f in path:
            pt = path[f]
            trail.append(pt)

        if len(trail) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in trail], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), 3)

        for af, apt in anchors.items():
            if af <= f:
                cv2.circle(vis, (int(apt[0]), int(apt[1])), 7, (0, 255, 0), -1)
                cv2.putText(vis, f"A{af}", (int(apt[0]) + 6, int(apt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if f in path:
            x, y = path[f]
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 255), 2)

        if args.debug and f in debug_candidates:
            for cand in debug_candidates[f]:
                gx = int(cand.x + x1)
                gy = int(cand.y + y1)
                cv2.circle(vis, (gx, gy), 3, (255, 255, 0), -1)
                cv2.putText(vis, f"{cand.score:.1f}", (gx + 3, gy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        out.write(vis)

        if args.display:
            disp = cv2.resize(vis, None, fx=args.resize_display, fy=args.resize_display)
            cv2.imshow("v10 full-path tracker", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    out.release()
    if args.display:
        cv2.destroyAllWindows()



def safe_id_from_video(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    clean = []
    for ch in base:
        if ch.isalnum() or ch in ["_", "-"]:
            clean.append(ch)
        else:
            clean.append("_")
    return "".join(clean).strip("_") or "pitch"


def ensure_header_csv(path: str, header: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def crop_with_padding(frame, cx: float, cy: float, size: int):
    h, w = frame.shape[:2]
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    out = np.zeros((size, size, 3), dtype=frame.dtype)

    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(w, x2)
    sy2 = min(h, y2)

    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if sx2 > sx1 and sy2 > sy1:
        out[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]

    return out


def save_dataset_outputs(args, frames, frame_numbers, fps, roi, path, anchors, metrics):
    dataset_dir = args.dataset_dir
    pitch_id = args.pitch_id or safe_id_from_video(args.video)

    labels_dir = os.path.join(dataset_dir, "labels")
    crops_dir = os.path.join(dataset_dir, "crops", pitch_id)
    paths_dir = os.path.join(dataset_dir, "paths")
    metrics_dir = os.path.join(dataset_dir, "metrics")

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    h, w = frames[0].shape[:2]
    x1, y1, x2, y2 = roi

    labels_path = os.path.join(labels_dir, f"{pitch_id}_frame_labels.csv")
    path_out = os.path.join(paths_dir, f"{pitch_id}_path.csv")
    metrics_out = os.path.join(metrics_dir, f"{pitch_id}_metrics.txt")

    manifest_csv = args.manifest_csv or os.path.join(dataset_dir, "train_manifest.csv")
    features_csv = args.features_csv or os.path.join(dataset_dir, "features.csv")

    frame_to_img = {f: img for f, img in zip(frame_numbers, frames)}

    label_header = [
        "pitch_id", "video_path", "pitch_type", "frame", "time_sec",
        "visible", "x", "y", "x_norm", "y_norm",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "x_roi", "y_roi", "anchor"
    ]

    manifest_header = [
        "image_path", "pitch_id", "video_path", "pitch_type", "frame",
        "visible", "x", "y", "x_norm", "y_norm", "anchor", "crop_size"
    ]

    with open(labels_path, "w", newline="") as lf, open(path_out, "w", newline="") as pf:
        lw = csv.writer(lf)
        pw = csv.writer(pf)
        lw.writerow(label_header)
        pw.writerow(["pitch_id", "pitch_type", "frame", "time_sec", "tracked", "x", "y", "anchor"])

        ensure_header_csv(manifest_csv, manifest_header)

        with open(manifest_csv, "a", newline="") as mf:
            mw = csv.writer(mf)

            for fr in frame_numbers:
                tracked = fr in path
                anchor = 1 if fr in anchors else 0
                t = fr / fps

                if tracked:
                    x, y = path[fr]
                    x_norm = x / max(w - 1, 1)
                    y_norm = y / max(h - 1, 1)
                    x_roi = x - x1
                    y_roi = y - y1
                    visible = 1

                    crop_name = f"{pitch_id}_frame_{fr:06d}.jpg"
                    crop_path = os.path.join(crops_dir, crop_name)
                    crop = crop_with_padding(frame_to_img[fr], x, y, args.crop_size)
                    cv2.imwrite(crop_path, crop)

                    mw.writerow([
                        crop_path, pitch_id, args.video, args.pitch_type, fr,
                        visible, f"{x:.3f}", f"{y:.3f}",
                        f"{x_norm:.6f}", f"{y_norm:.6f}",
                        anchor, args.crop_size
                    ])

                    lw.writerow([
                        pitch_id, args.video, args.pitch_type, fr, f"{t:.6f}",
                        visible, f"{x:.3f}", f"{y:.3f}",
                        f"{x_norm:.6f}", f"{y_norm:.6f}",
                        x1, y1, x2, y2,
                        f"{x_roi:.3f}", f"{y_roi:.3f}", anchor
                    ])
                    pw.writerow([pitch_id, args.pitch_type, fr, f"{t:.6f}", 1, f"{x:.3f}", f"{y:.3f}", anchor])
                else:
                    lw.writerow([
                        pitch_id, args.video, args.pitch_type, fr, f"{t:.6f}",
                        0, "", "", "", "",
                        x1, y1, x2, y2,
                        "", "", anchor
                    ])
                    pw.writerow([pitch_id, args.pitch_type, fr, f"{t:.6f}", 0, "", "", anchor])

    with open(metrics_out, "w") as f:
        f.write(f"pitch_id: {pitch_id}\\n")
        f.write(f"video_path: {args.video}\\n")
        f.write(f"pitch_type: {args.pitch_type}\\n")
        f.write(f"roi: {roi}\\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\\n")

    feature_keys = [
        "status", "num_points", "first_frame", "last_frame", "duration_sec",
        "avg_speed_px_per_frame", "max_speed_px_per_frame", "path_length_px",
        "direct_distance_px", "straightness_ratio", "max_perpendicular_deviation_px",
        "quadratic_curvature_y_of_x", "anchor_count", "heuristic_label"
    ]
    feature_header = ["pitch_id", "video_path", "pitch_type", "roi_x1", "roi_y1", "roi_x2", "roi_y2"] + feature_keys
    ensure_header_csv(features_csv, feature_header)

    with open(features_csv, "a", newline="") as ff:
        writer = csv.writer(ff)
        row = [pitch_id, args.video, args.pitch_type, x1, y1, x2, y2]
        for k in feature_keys:
            row.append(metrics.get(k, ""))
        writer.writerow(row)

    print("[DATASET] Saved dataset outputs:")
    print(f"  labels:   {labels_path}")
    print(f"  path:     {path_out}")
    print(f"  metrics:  {metrics_out}")
    print(f"  manifest: {manifest_csv}")
    print(f"  features: {features_csv}")
    print(f"  crops:    {crops_dir}")

def main():
    args = parse_args()
    if args.pitch_id is None:
        args.pitch_id = safe_id_from_video(args.video)

    print("[INFO] Loading video...")
    frames, frame_numbers, fps, total = read_video(args.video, args.start_frame, args.end_frame)
    if not frames:
        raise RuntimeError("No frames loaded. Check start/end frame and video path.")

    print(f"[INFO] Loaded {len(frames)} frames from total={total}, fps={fps:.3f}")
    first_frame = frames[0]

    if args.select_roi or args.roi is None:
        roi = choose_roi(first_frame)
    else:
        roi = tuple(args.roi)

    x1, y1, x2, y2 = roi
    print(f"[INFO] ROI = {roi}")

    # Anchors
    anchors: Dict[int, Point] = {}
    if args.anchor_file and os.path.exists(args.anchor_file):
        anchors = load_anchor_file(args.anchor_file)
        print(f"[INFO] Loaded {len(anchors)} anchors from {args.anchor_file}")
    else:
        frame_to_img = {f: img for f, img in zip(frame_numbers, frames)}
        for af in args.manual_anchor_frames:
            if af not in frame_to_img:
                print(f"[WARN] Anchor frame {af} is outside loaded frame window, skipping")
                continue
            pt = click_anchor(frame_to_img[af], af, roi=roi, resize=args.resize_display)
            if pt is not None:
                anchors[af] = pt
                print(f"[INFO] Added manual anchor: frame={af}, x={pt[0]:.1f}, y={pt[1]:.1f}")
        if anchors:
            auto_anchor_file = os.path.splitext(args.csv)[0] + "_anchors.csv"
            save_anchor_file(auto_anchor_file, anchors)
            print(f"[INFO] Saved anchors to {auto_anchor_file}")

    if len(anchors) < 2:
        print("[WARN] You should use at least 2 anchors, ideally 5-10 across the full ball flight.")

    print("[INFO] Building ROI frames and evidence maps...")
    roi_frames = [crop_roi(fr, roi) for fr in frames]
    grays, hsvs, motion_maps = build_evidence_maps(roi_frames, args.motion_thresh)

    print("[INFO] Building anchor-guided full trajectory...")
    path, debug_candidates = build_path(frame_numbers, roi_frames, grays, hsvs, motion_maps, anchors, roi, args)

    print(f"[INFO] Path points: {len(path)}")

    #CSV
    print(f"[INFO] Writing CSV: {args.csv}")
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_sec", "tracked", "x", "y", "anchor"])
        for fr in frame_numbers:
            if fr in path:
                x, y = path[fr]
                writer.writerow([fr, fr / fps, 1, f"{x:.3f}", f"{y:.3f}", 1 if fr in anchors else 0])
            else:
                writer.writerow([fr, fr / fps, 0, "", "", 1 if fr in anchors else 0])

    metrics = compute_metrics(path, fps, anchors)
    print(f"[INFO] Writing metrics: {args.metrics}")
    with open(args.metrics, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    if args.save_dataset:
        save_dataset_outputs(args, frames, frame_numbers, fps, roi, path, anchors, metrics)

    print(f"[INFO] Writing output video: {args.output}")
    draw_output(frames, frame_numbers, roi, path, anchors, debug_candidates, args, fps)

    print("[DONE]")
    print("[METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
