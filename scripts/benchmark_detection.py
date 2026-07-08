"""
Runs the OLD (pre-Phase-1) and NEW detection/tracking configurations against a
set of sample videos and prints/saves a comparison table of proxy metrics.

There is no labeled ground truth for this repo's videos, so "better" is judged
via proxies computed from the pipeline's own output structures:

  - ball_continuity_pct: fraction of frames backed by a real (non-interpolated/
    non-predicted) ball detection. Expected to improve because the lower ball
    confidence threshold (configs.BALL_CONF_THRESHOLD) recovers real detections
    the old fixed 0.5 threshold discarded, while the Kalman gate in
    BallTracker.get_object_tracks_with_kalman rejects the false positives that
    lower threshold also lets through -- net effect should be more real signal,
    not more noise.
  - ball_trajectory_smoothness: sum of squared second-differences of the ball's
    center position across consecutive frames. Lower = more physically
    plausible motion (a basketball doesn't teleport); tests whether
    Kalman-bridged gaps produce more plausible arcs than the old pandas linear
    interpolation.
  - team_flip_count: total team-assignment transitions across all players
    across all frames. The old TeamAssigner reset its player->team mapping
    every 50 frames as a workaround for CLIP's per-call instability -- a
    structural source of spurious flips independent of any real jersey-color
    ambiguity. The new k-means + rolling-majority-vote TeamAssigner has no such
    reset, so flip count should drop toward 0.
  - track_id_churn_pct: fraction of distinct player track IDs that appear in
    fewer than `--churn-min-frames` frames -- a proxy for tracking
    fragmentation from occlusion-induced ID loss. Lower is better; the new
    ByteTrack tuning (correct frame_rate, larger lost_track_buffer) targets
    this directly.
  - stage_timings_sec: wall-clock time per stage, reusing the same
    perf_counter() pattern main.py's --profile flag already uses.

Requires the 3 pretrained YOLO model weights (models/*.pt) to actually run --
see README.md's Installation section for the Google Drive download links. If
the weights aren't present yet, this script exits early with a clear message
rather than a confusing torch/ultralytics stack trace.

Usage:
    python scripts/benchmark_detection.py
    python scripts/benchmark_detection.py --videos input_videos/video_1.mp4
    python scripts/benchmark_detection.py --output results/benchmark.json
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal.basketball_analysis.utils import read_video, get_video_fps
from personal.basketball_analysis.trackers import PlayerTracker, BallTracker
from personal.basketball_analysis.team_assigner import TeamAssigner
from personal.basketball_analysis.team_assigner.team_assigner_legacy import LegacyTeamAssigner
from personal.basketball_analysis.configs import (
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    PLAYER_CONF_THRESHOLD,
    BALL_CONF_THRESHOLD,
    PLAYER_AUGMENT,
    BALL_AUGMENT,
    BALL_KF_PROCESS_NOISE,
    BALL_KF_MEASUREMENT_NOISE,
    BALL_KF_MAX_REASONABLE_DISTANCE_PER_FRAME,
    PLAYER_TRACK_ACTIVATION_THRESHOLD,
    PLAYER_LOST_TRACK_BUFFER,
    PLAYER_MINIMUM_MATCHING_THRESHOLD,
    PLAYER_MINIMUM_CONSECUTIVE_FRAMES,
)

DEFAULT_VIDEOS = [
    "input_videos/video_1.mp4",
    "input_videos/video_2.mp4",
    "input_videos/video_3.mp4",
]

# The exact defaults sv.ByteTrack() and the old BallTracker/TeamAssigner used
# before Phase 1 -- reproduced here (not hardcoded elsewhere) so the "old"
# variant is an honest baseline, not a strawman.
OLD_BYTETRACK_DEFAULTS = dict(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    minimum_consecutive_frames=1,
)
OLD_SHARED_CONF = 0.5


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compute_ball_continuity_from_source_field(tracks):
    """New pipeline: fraction of frames whose ball entry came from a real detection."""
    if not tracks:
        return 0.0
    n_detected = sum(1 for t in tracks if t.get(1, {}).get("source") == "detection")
    return n_detected / len(tracks)


def compute_ball_continuity_from_presence(tracks):
    """Old pipeline: fraction of frames with a real ball bbox BEFORE interpolation fills gaps."""
    if not tracks:
        return 0.0
    n_present = sum(1 for t in tracks if t.get(1, {}).get("bbox"))
    return n_present / len(tracks)


def compute_trajectory_smoothness(final_ball_tracks):
    """Sum of squared second-differences of ball center position. Lower = smoother/more plausible."""
    centers = []
    for t in final_ball_tracks:
        bbox = t.get(1, {}).get("bbox")
        if bbox:
            centers.append(_bbox_center(bbox))
    if len(centers) < 3:
        return float("nan")
    second_diffs = []
    for i in range(1, len(centers) - 1):
        dx = centers[i + 1][0] - 2 * centers[i][0] + centers[i - 1][0]
        dy = centers[i + 1][1] - 2 * centers[i][1] + centers[i - 1][1]
        second_diffs.append(dx * dx + dy * dy)
    return sum(second_diffs)


def compute_team_flip_count(player_assignment):
    """Total team-id transitions across all players across all frames. Lower is better."""
    history = defaultdict(list)
    for frame in player_assignment:
        for player_id, team_id in frame.items():
            history[player_id].append(team_id)
    flips = 0
    for sequence in history.values():
        flips += sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    return flips


def compute_track_id_churn(player_tracks, min_frames=10):
    """Fraction of distinct track IDs appearing in fewer than min_frames frames. Lower is better."""
    id_counts = Counter()
    for frame in player_tracks:
        for track_id in frame:
            id_counts[track_id] += 1
    if not id_counts:
        return 0.0
    short_lived = sum(1 for count in id_counts.values() if count < min_frames)
    return short_lived / len(id_counts)


def _check_models_present():
    missing = [p for p in (PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH) if not os.path.exists(p)]
    if missing:
        print(
            "ERROR: missing model weight file(s): " + ", ".join(missing) + "\n"
            "This benchmark needs the pretrained YOLO weights to run real inference.\n"
            "See README.md's Installation section for the Google Drive download\n"
            "links, and place the .pt files in models/.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_old_variant(video_path, video_frames, source_fps, churn_min_frames):
    timings = {}

    t0 = time.perf_counter()
    player_tracker = PlayerTracker(
        PLAYER_DETECTOR_PATH, conf=OLD_SHARED_CONF, video_fps=30.0,  # old code never passed real fps
        **OLD_BYTETRACK_DEFAULTS,
    )
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None)
    timings["player_tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ball_tracker = BallTracker(BALL_DETECTOR_PATH, conf=OLD_SHARED_CONF)
    raw_ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None)
    pre_interpolation_tracks = ball_tracker.remove_wrong_detections(list(raw_ball_tracks))
    final_ball_tracks = ball_tracker.interpolate_ball_positions(list(pre_interpolation_tracks))
    timings["ball_tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    team_assigner = LegacyTeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames, player_tracks, read_from_stub=False, stub_path=None
    )
    timings["team_assignment"] = time.perf_counter() - t0

    return {
        "ball_continuity_pct": round(100 * compute_ball_continuity_from_presence(pre_interpolation_tracks), 2),
        "ball_trajectory_smoothness": compute_trajectory_smoothness(final_ball_tracks),
        "team_flip_count": compute_team_flip_count(player_assignment),
        "track_id_churn_pct": round(100 * compute_track_id_churn(player_tracks, churn_min_frames), 2),
        "stage_timings_sec": {k: round(v, 3) for k, v in timings.items()},
    }


def run_new_variant(video_path, video_frames, source_fps, churn_min_frames):
    timings = {}

    t0 = time.perf_counter()
    player_tracker = PlayerTracker(
        PLAYER_DETECTOR_PATH, conf=PLAYER_CONF_THRESHOLD, augment=PLAYER_AUGMENT,
        video_fps=source_fps,
        track_activation_threshold=PLAYER_TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=PLAYER_LOST_TRACK_BUFFER,
        minimum_matching_threshold=PLAYER_MINIMUM_MATCHING_THRESHOLD,
        minimum_consecutive_frames=PLAYER_MINIMUM_CONSECUTIVE_FRAMES,
    )
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None)
    timings["player_tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ball_tracker = BallTracker(
        BALL_DETECTOR_PATH, conf=BALL_CONF_THRESHOLD, augment=BALL_AUGMENT,
        kf_process_noise=BALL_KF_PROCESS_NOISE,
        kf_measurement_noise=BALL_KF_MEASUREMENT_NOISE,
        max_reasonable_distance_per_frame=BALL_KF_MAX_REASONABLE_DISTANCE_PER_FRAME,
    )
    final_ball_tracks = ball_tracker.get_object_tracks_with_kalman(video_frames, read_from_stub=False, stub_path=None)
    timings["ball_tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames, player_tracks, read_from_stub=False, stub_path=None
    )
    timings["team_assignment"] = time.perf_counter() - t0

    return {
        "ball_continuity_pct": round(100 * compute_ball_continuity_from_source_field(final_ball_tracks), 2),
        "ball_trajectory_smoothness": compute_trajectory_smoothness(final_ball_tracks),
        "team_flip_count": compute_team_flip_count(player_assignment),
        "track_id_churn_pct": round(100 * compute_track_id_churn(player_tracks, churn_min_frames), 2),
        "stage_timings_sec": {k: round(v, 3) for k, v in timings.items()},
    }


def _fmt(value):
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return str(value)


def _print_comparison_table(results):
    print("\n" + "=" * 78)
    print("BENCHMARK: old (pre-Phase-1) vs new detection/tracking pipeline")
    print("=" * 78)
    metric_keys = [
        ("ball_continuity_pct", "Ball continuity % (higher=better)"),
        ("ball_trajectory_smoothness", "Ball trajectory smoothness (lower=better)"),
        ("team_flip_count", "Team-assignment flips (lower=better)"),
        ("track_id_churn_pct", "Track ID churn % (lower=better)"),
    ]
    for video, variants in results.items():
        print(f"\n{video}")
        print(f"  {'metric':42s} {'old':>12s} {'new':>12s}")
        for key, label in metric_keys:
            old_val = variants["old"][key]
            new_val = variants["new"][key]
            print(f"  {label:42s} {_fmt(old_val):>12s} {_fmt(new_val):>12s}")
        print(f"  {'--- stage timings (sec) ---':42s}")
        for stage in variants["old"]["stage_timings_sec"]:
            old_t = variants["old"]["stage_timings_sec"].get(stage, float("nan"))
            new_t = variants["new"]["stage_timings_sec"].get(stage, float("nan"))
            print(f"  {stage:42s} {old_t:>12.3f} {new_t:>12.3f}")
    print("\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS,
                        help="Video paths to benchmark against")
    parser.add_argument("--output", default="results/benchmark.json",
                        help="Where to write the JSON results")
    parser.add_argument("--churn-min-frames", type=int, default=10,
                        help="Track ID lifespan (frames) below which a track counts as churn")
    args = parser.parse_args()

    _check_models_present()

    results = {}
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"WARNING: {video_path} not found, skipping.", file=sys.stderr)
            continue
        print(f"Benchmarking {video_path} ...")
        video_frames = read_video(video_path)
        source_fps = get_video_fps(video_path)

        old_result = run_old_variant(video_path, video_frames, source_fps, args.churn_min_frames)
        new_result = run_new_variant(video_path, video_frames, source_fps, args.churn_min_frames)
        results[video_path] = {"old": old_result, "new": new_result}

    if not results:
        print("No videos were benchmarked (none found on disk).", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {output_path}")

    _print_comparison_table(results)


if __name__ == "__main__":
    main()
